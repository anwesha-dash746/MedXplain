/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // The browser talks to /api/* on the same origin; Next proxies to FastAPI so the
  // demo works without CORS configuration or a second public hostname.
  async rewrites() {
    const backend = process.env.MEDXPLAIN_API_URL || "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};
export default nextConfig;
