import argparse
import asyncio
import json, logging, os, ssl, uuid
import numpy as np
import cv2, torch
import time

from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

m_channel = dict()
mask_model = torch.hub.load("ultralytics/yolov5", "custom", path="v5_1017_1_best.pt", force_reload=True)

def plot_boxes(results, frame):
    for i in range(len(results)):
        row = results.loc[i, :].values.tolist()
        if row[4] >= 0.30:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            bgr = (0, 255, 0)
            if row[5] == 1:
                bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, row[6], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, transform, uuid):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.results = []
        self.framecnt = 0
        self.uuid = uuid

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "mask":
            img = frame.to_ndarray(format="bgr24")#, width=640, height=320)

            if self.framecnt % 6 == 0:
                self.results = mask_model(img, size=640)
                self.framecnt = 0
            self.framecnt += 1

            img = plot_boxes(self.results.pandas().xyxy[-1], img)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    uid = uuid.uuid4()
    pc_id = "PeerConnection(%s)" % uid
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        global m_channel
        m_channel[uid] = channel

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"], uuid=uid
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
