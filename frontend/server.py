#!/usr/bin/env python3
"""
Простой HTTP сервер для локального тестирования фронтенда.

Использование:
    python server.py [port]
    
По умолчанию запускается на http://localhost:3000

Особенности:
- Поддержка CORS для WebSocket подключений
- Правильные MIME types для JS модулей
- SharedArrayBuffer headers для FFmpeg WASM
"""

import http.server
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler с поддержкой CORS и правильными headers"""
    
    extensions_map = {
        '': 'application/octet-stream',
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.wasm': 'application/wasm',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.webm': 'audio/webm',
    }
    
    def end_headers(self):
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # Headers для SharedArrayBuffer (нужен для FFmpeg WASM)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        
        # Cache control
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Цветной лог"""
        status = args[1] if len(args) > 1 else ''
        
        if status.startswith('2'):
            color = '\033[92m'  # Green
        elif status.startswith('3'):
            color = '\033[93m'  # Yellow
        elif status.startswith('4') or status.startswith('5'):
            color = '\033[91m'  # Red
        else:
            color = '\033[0m'   # Default
        
        reset = '\033[0m'
        print(f"{color}[{self.log_date_time_string()}] {format % args}{reset}")


def main():
    # Переходим в директорию frontend
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║                  Parakeet ASR Frontend                     ║
╠════════════════════════════════════════════════════════════╣
║  Сервер запущен: http://localhost:{PORT:<24}║
║                                                            ║
║  Нажмите Ctrl+C для остановки                              ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nСервер остановлен.")
            sys.exit(0)


if __name__ == "__main__":
    main()
