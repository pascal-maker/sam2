# Stage 1: Build Stage
FROM node:22.9.0 AS build

WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

RUN groupadd --system frontend \
    && useradd --system --gid frontend --home-dir /app --shell /usr/sbin/nologin frontend \
    && chown -R frontend:frontend /app
USER frontend

# Copy source code
COPY --chown=frontend:frontend . .

# Build the application
RUN yarn build

# Stage 2: Production Stage
FROM nginx:latest

RUN groupadd --system frontend \
    && useradd --system --gid frontend --home-dir /usr/share/nginx/html --shell /usr/sbin/nologin frontend \
    && sed -i 's/listen       80;/listen       8080;/' /etc/nginx/conf.d/default.conf \
    && sed -i 's/listen  \\[::\\]:80;/listen  [::]:8080;/' /etc/nginx/conf.d/default.conf \
    && chown -R frontend:frontend /usr/share/nginx/html /var/cache/nginx /var/log/nginx /etc/nginx/conf.d \
    && touch /var/run/nginx.pid \
    && chown frontend:frontend /var/run/nginx.pid

# Copy built files from the build stage to the production image
COPY --from=build --chown=frontend:frontend /app/dist /usr/share/nginx/html

USER frontend
EXPOSE 8080

# Container startup command for the web server (nginx in this case)
CMD ["nginx", "-g", "daemon off;"]
