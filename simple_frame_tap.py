#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_frame_tap.py
-------------------
Read from a serial port and ONLY display data immediately following a provided header.
Two modes:
  1) --after N        -> print the next N bytes after each header occurrence
  2) --frame-len L    -> print the entire frame of length L starting at the header (header included)

Header is given as hex bytes, e.g. "55 61" or "0x55,0x61".
Output is hex by default. Use --ascii to dump ASCII for the printed region.

Examples:
  # Print the 6 bytes after each '55 61' header at 460800 baud
  python simple_frame_tap.py --port COM5 --baud 460800 --header "55 61" --after 6

  # Print whole 33-byte frames that start with '55 61'
  python simple_frame_tap.py --port COM5 --baud 460800 --header "55 61" --frame-len 33
"""
import argparse
import sys

try:
    import serial
except Exception:
    print("pyserial is required. Install with: pip install pyserial", file=sys.stderr)
    raise

def parse_hex_bytes(s: str) -> bytes:
    toks = []
    token = ""
    for ch in s:
        if ch in "0123456789abcdefABCDEFxX":
            token += ch
        else:
            if token:
                toks.append(token)
                token = ""
    if token:
        toks.append(token)
    out = []
    for t in toks:
        t = t.strip()
        if t.lower().startswith("0x"):
            t = t[2:]
        if len(t) == 1:
            t = "0" + t
        if len(t) != 2:
            raise ValueError(f"Invalid hex byte: {t}")
        out.append(int(t, 16))
    return bytes(out)

def hexdump(b: bytes) -> str:
    return " ".join(f"{x:02X}" for x in b)

def main():
    ap = argparse.ArgumentParser(description="Tap and print bytes following a specific header.")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--header", required=True, help='Header bytes, e.g. "55 61"')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--after", type=int, help="Print the next N bytes after the header (header not included)")
    g.add_argument("--frame-len", type=int, help="Print a whole frame of length L starting at header (header included)")
    ap.add_argument("--ascii", action="store_true", help="Print as ASCII instead of hex")
    args = ap.parse_args()

    header = parse_hex_bytes(args.header)
    if len(header) == 0:
        print("Header must not be empty.", file=sys.stderr)
        sys.exit(1)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.02)
    except Exception as e:
        print(f"Failed to open {args.port}: {e}", file=sys.stderr)
        sys.exit(2)

    buf = bytearray()
    printed = 0

    try:
        while True:
            chunk = ser.read(2048)
            if chunk:
                buf.extend(chunk)

            i = buf.find(header)
            if i < 0:
                keep = max(0, len(header) - 1)
                if len(buf) > keep:
                    del buf[:len(buf) - keep]
                continue

            if args.frame_len is not None:
                end = i + args.frame_len
                if end <= len(buf):
                    frame = bytes(buf[i:end])
                    del buf[:end]
                    if args.ascii:
                        try:
                            print(frame.decode('utf-8', errors='replace'))
                        except Exception:
                            print(frame)
                    else:
                        print(hexdump(frame))
                    printed += 1
                else:
                    if i > 0:
                        del buf[:i]
                continue

            if args.after is not None:
                start = i + len(header)
                end = start + args.after
                if end <= len(buf):
                    data = bytes(buf[start:end])
                    del buf[:end]
                    if args.ascii:
                        try:
                            print(data.decode('utf-8', errors='replace'))
                        except Exception:
                            print(data)
                    else:
                        print(hexdump(data))
                    printed += 1
                else:
                    if i > 0:
                        del buf[:i]
                continue

    except KeyboardInterrupt:
        print(f"\nDone. Printed {printed} segment(s).")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
