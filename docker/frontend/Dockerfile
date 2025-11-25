FROM node:18-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache libc6-compat

# 复制package.json和package-lock.json
COPY package*.json ./

# 安装依赖
RUN npm install --ignore-scripts

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 3000

# 启动命令
CMD ["npm", "start"]