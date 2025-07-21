docker build -t my-gevent-app .

docker run -d --restart always -p 5002:5001 --env-file .env --name my-app-instance my-gevent-app