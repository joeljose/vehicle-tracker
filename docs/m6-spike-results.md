# M6 Spike Results: nvurisrcbin + YouTube HLS

**Date:** 2026-03-31
**Issue:** #79
**Stream tested:** `https://www.youtube.com/watch?v=eAdWk0Vg64A` (741 & Lytle South — target junction)

---

## 1. Can nvurisrcbin consume a YouTube HLS URL?

**YES.** Works out of the box.

**Setup:**
- `yt-dlp -f "bv*+ba/b" --get-url <youtube_url>` resolves to an HLS manifest URL
- `nvurisrcbin uri=<hls_url>` consumes it natively via `GstSoupHTTPSrc` + HLS demux
- Dynamic pad (`vsrc_0`) links to downstream elements

**Results:**
- 90 frames received in ~3 seconds (first test)
- 748 frames in ~10 seconds at steady state (~75 fps burst, 30 fps source)
- Stream: 1920x1080, H.264, 30 fps, ~5421 kbps (best quality auto-selected)
- Audio warning expected and harmless: `No decoder available for type 'audio/mpeg'`

**No special GStreamer properties needed.** Default `nvurisrcbin` settings work.

---

## 2. GStreamer error messages on HLS failure

Tested with an invalid/expired HLS URL. Three error messages emitted:

| # | Source | Domain | Code | Message |
|---|--------|--------|------|---------|
| 1 | `source` (GstSoupHTTPSrc) | `gst-resource-error-quark` | 15 | `Forbidden` (HTTP 403) |
| 2 | `source` (GstSoupHTTPSrc) | `gst-stream-error-quark` | 1 | `Internal data stream error` (streaming stopped, reason error -5) |
| 3 | `typefindelement0` | `gst-stream-error-quark` | 4 | `Stream doesn't contain enough data` |

**Key finding:** The first error (`gst-resource-error-quark`, code 15) is the actionable one. It fires immediately on HTTP 403 (expired URL). The other two are cascading failures.

**Detection strategy:** Listen for `GST_MESSAGE_ERROR` on the bus. Match `gst-resource-error-quark` errors from the `nvurisrcbin` subtree to identify stream failures. The source element name contains the `nvurisrcbin` name prefix.

---

## 3. Frame-gap watchdog

**Works.** The watchdog (1-second polling interval, 3-second threshold) detected a natural HLS buffering gap during the steady-state test:

```
WATCHDOG: No frames for 3.9s (last frame at count=748)
```

This was a natural HLS segment boundary gap, not a stream failure. For production:
- **Recommendation:** Use a longer threshold (5-10 seconds) to avoid false positives from HLS buffering
- The GStreamer error messages are the primary detection mechanism
- Watchdog is the fallback for silent freezes (no error emitted)

---

## 4. Stream metadata

```
Title:    Live Traffic @ 741 & Lytle South 2026-03-31 08:52
Channel:  City of Springboro Live Traffic
Is live:  True

Available HLS formats:
  240p  - 426x240   avc1.4D4015  30fps   546 kbps
  360p  - 640x360   avc1.4D401E  30fps  1210 kbps
  480p  - 854x480   avc1.4D401F  30fps  1569 kbps
  720p  - 1280x720  avc1.4D401F  30fps  2969 kbps
  1080p - 1920x1080 avc1.640028  30fps  5421 kbps
```

---

## 5. Conclusion

**No blockers.** All three validation questions answered positively:

1. `nvurisrcbin` handles YouTube HLS natively -- no alternative approach needed
2. GStreamer bus errors are clear and actionable (HTTP 403 on expired URL)
3. Frame-gap watchdog works as fallback (use 5-10s threshold to avoid HLS buffering false positives)

**Proceed to P1 implementation.**
