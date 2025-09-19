# Use lightweight nginx to serve React build
FROM nginx:alpine

# Copy build files into nginx html folder
COPY build /usr/share/nginx/html

# Replace default nginx config to support React Router SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]