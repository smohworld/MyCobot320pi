#!/usr/bin/env python3
"""Run this ON the myCobot 320 Pi (no ROS needed, only pyserial).

It forwards a TCP connection byte-for-byte to the robot's internal serial port,
so the PC can use pymycobot.MyCobot320Socket(ip, 9000) / the ROS driver with
connection:=socket. One client at a time; a new client is served after the
previous one disconnects.

  python3 mycobot320_socket_server.py                 # /dev/ttyAMA0, 115200, port 9000
  python3 mycobot320_socket_server.py --port 9000 --serial /dev/ttyAMA0

Stop other programs that use the serial port (myBlockly, other scripts) first.
"""
import argparse
import socket
import threading

import serial


def serve(conn, ser):
    stop = threading.Event()

    def serial_to_net():
        while not stop.is_set():
            try:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    conn.sendall(data)
            except (OSError, serial.SerialException):
                break
        stop.set()

    t = threading.Thread(target=serial_to_net, daemon=True)
    t.start()
    try:
        while not stop.is_set():
            data = conn.recv(1024)
            if not data:
                break
            ser.write(data)
    except OSError:
        pass
    finally:
        stop.set()
        t.join(1.0)
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--serial', default='/dev/ttyAMA0')
    ap.add_argument('--baud', type=int, default=115200)
    a = ap.parse_args()

    ser = serial.Serial(a.serial, a.baud, timeout=0.05)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port))
    srv.listen(1)
    print(f'serving {a.serial}@{a.baud} on tcp port {a.port}', flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f'client {addr} connected', flush=True)
        ser.reset_input_buffer()
        serve(conn, ser)
        print(f'client {addr} disconnected', flush=True)


if __name__ == '__main__':
    main()
