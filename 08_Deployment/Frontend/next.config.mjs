/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  serverExternalPackages: ['better-sqlite3'],
  typescript: {
    ignoreBuildErrors: true,
  }
}

export default nextConfig
