/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  serverExternalPackages: ['better-sqlite3'],
  typescript: {
    ignoreBuildErrors: true,
  }
}

export default nextConfig
