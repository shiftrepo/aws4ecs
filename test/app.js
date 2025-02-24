const http = require('http');

const hostname = '0.0.0.0'; // Fargate環境では 0.0.0.0 を使用
const port = 80; // または 8080 など、任意のポート

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  res.end('Hello, Fargate from Docker Hub!\n');
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://<span class="math-inline">\{hostname\}\:</span>{port}/`);
});

