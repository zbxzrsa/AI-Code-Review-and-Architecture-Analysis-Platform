FROM node:18-alpine

WORKDIR /app

# 安装依赖
COPY package*.json ./
RUN npm install

# 复制应用代码
COPY . .

# 设置环境变量
ENV NODE_ENV=production
ENV PORT=8000
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV CONSUL_HOST=consul
ENV CONSUL_PORT=8500
ENV SERVICE_NAME=api-gateway
ENV JWT_SECRET=your_jwt_secret_key_here
ENV RATE_LIMIT_MAX=100
ENV RATE_LIMIT_WINDOW_MS=60000

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["node", "server.js"]