package com.chinesechess.engine

import java.net.*
import java.util.concurrent.ConcurrentLinkedQueue

class AndroidNetworkClient : NetworkClient {
    override var myIp = "127.0.0.1"
        private set
    override var isOnline = false
        private set
    override var myOnlineSide = PSide.RED
    override val discoveredUsers: List<OnlineUser>
        get() = _discoveredUsers.toList()

    private val _discoveredUsers = ConcurrentLinkedQueue<OnlineUser>()
    private var udpSock: DatagramSocket? = null
    private var tcpListenSock: ServerSocket? = null
    private var tcpConnSock: Socket? = null
    private var tcpPort = 0
    private var lastBroadcast = 0L
    private var userName = ""
    private var active = false
    private var incomingInvite: InviteInfo? = null
    private val moveQueue = ConcurrentLinkedQueue<Move>()
    private var inviteSent = false

    override fun init(userName: String) {
        if (active) return
        this.userName = userName
        myIp = discoverMyIp()
        _discoveredUsers.clear()
        udpSock = DatagramSocket(9999).apply { broadcast = true; soTimeout = 500 }
        tcpListenSock = ServerSocket(0).apply { soTimeout = 500 }
        tcpPort = tcpListenSock!!.localPort
        active = true
        isOnline = false
        inviteSent = false
        incomingInvite = null
    }

    override fun cleanup() {
        active = false
        isOnline = false
        try { udpSock?.close() } catch (_: Exception) {}
        try { tcpListenSock?.close() } catch (_: Exception) {}
        try { tcpConnSock?.close() } catch (_: Exception) {}
        udpSock = null
        tcpListenSock = null
        tcpConnSock = null
    }

    override fun broadcast() {
        if (!active) return
        val now = System.currentTimeMillis() / 1000
        if (now - lastBroadcast < 2) return
        lastBroadcast = now
        try {
            val sock = DatagramSocket()
            sock.broadcast = true
            val msg = "CHESS_HELLO $userName $tcpPort"
            val data = msg.toByteArray()
            val addr = InetAddress.getByName("255.255.255.255")
            sock.send(DatagramPacket(data, data.size, addr, 9999))
            sock.close()
        } catch (_: Exception) {}
    }

    override fun poll(): InviteInfo? {
        if (!active) return null
        broadcast()
        receiveUdp()
        acceptTcp()
        readTcp()
        return incomingInvite?.also { incomingInvite = null }
    }

    private fun receiveUdp() {
        val sock = udpSock ?: return
        val buf = ByteArray(512)
        val packet = DatagramPacket(buf, buf.size)
        try {
            while (true) {
                sock.receive(packet)
                val msg = String(packet.data, packet.offset, packet.length)
                if (msg.startsWith("CHESS_HELLO")) {
                    val parts = msg.split(" ")
                    if (parts.size >= 3) {
                        val name = parts[1]
                        val port = parts[2].toIntOrNull() ?: 0
                        val ip = packet.address.hostAddress ?: ""
                        if (name != userName) {
                            val now = System.currentTimeMillis()
                            _discoveredUsers.removeAll { it.ip == ip && it.port == port }
                            _discoveredUsers.add(OnlineUser(name, ip, port, now))
                            sendInviteInternal(ip, port, PSide.RED)
                        }
                    }
                } else if (msg.startsWith("CHESS_INVITE")) {
                    val parts = msg.split(" ")
                    if (parts.size >= 4) {
                        incomingInvite = InviteInfo(
                            fromName = parts[1],
                            fromIp = packet.address.hostAddress ?: "",
                            fromPort = parts[2].toIntOrNull() ?: 0,
                            fromSide = if (parts[3] == "1") PSide.RED else PSide.BLACK
                        )
                    }
                }
            }
        } catch (_: SocketTimeoutException) {
        } catch (_: Exception) {}
    }

    private fun sendInviteInternal(ip: String, port: Int, side: PSide) {
        if (inviteSent) return
        try {
            val sock = Socket()
            sock.connect(InetSocketAddress(ip, port), 1000)
            val msg = "CHESS_INVITE $userName $tcpPort ${if (side == PSide.RED) 1 else 2}"
            sock.getOutputStream().write(msg.toByteArray())
            sock.getOutputStream().flush()
            sock.close()
            inviteSent = true
        } catch (_: Exception) {}
    }

    override fun sendInvite(ip: String, port: Int, side: PSide) {
        sendInviteInternal(ip, port, side)
    }

    override fun acceptInvite(side: PSide) {
        myOnlineSide = side
        isOnline = true
    }

    override fun refreshUsers() {
        inviteSent = false
        _discoveredUsers.clear()
        broadcast()
        receiveUdp()
    }

    private fun acceptTcp() {
        val listen = tcpListenSock ?: return
        try {
            val conn = listen.accept()
            tcpConnSock = conn
            conn.soTimeout = 500
            isOnline = true
        } catch (_: SocketTimeoutException) {
        } catch (_: Exception) {}
    }

    private fun readTcp() {
        val conn = tcpConnSock ?: return
        try {
            val buf = ByteArray(256)
            val n = conn.getInputStream().read(buf)
            if (n > 0) {
                val msg = String(buf, 0, n)
                if (msg.startsWith("MOVE")) {
                    val parts = msg.split(" ")
                    if (parts.size >= 5) {
                        val fx = parts[1].toIntOrNull() ?: return
                        val fy = parts[2].toIntOrNull() ?: return
                        val tx = parts[3].toIntOrNull() ?: return
                        val ty = parts[4].toIntOrNull() ?: return
                        moveQueue.add(Move(fx, fy, tx, ty))
                    }
                }
            }
        } catch (_: SocketTimeoutException) {
        } catch (_: Exception) {}
    }

    override fun recvMove(): Move? = moveQueue.poll()

    override fun sendMove(move: Move) {
        val conn = tcpConnSock ?: return
        try {
            val msg = "MOVE ${move.fromRow} ${move.fromCol} ${move.toRow} ${move.toCol}"
            conn.getOutputStream().write(msg.toByteArray())
            conn.getOutputStream().flush()
        } catch (_: Exception) {}
    }

    private fun discoverMyIp(): String {
        return try {
            NetworkInterface.getNetworkInterfaces()?.toList()?.flatMap { iface ->
                iface.inetAddresses.toList().filter { addr ->
                    !addr.isLoopbackAddress && addr is Inet4Address
                }
            }?.firstOrNull()?.hostAddress ?: try {
                val sock = DatagramSocket()
                sock.connect(InetSocketAddress("8.8.8.8", 80))
                sock.localAddress.hostAddress ?: "127.0.0.1"
            } catch (_: Exception) { "127.0.0.1" }
        } catch (_: Exception) {
            try {
                val sock = DatagramSocket()
                sock.connect(InetSocketAddress("8.8.8.8", 80))
                sock.localAddress.hostAddress ?: "127.0.0.1"
            } catch (_: Exception) { "127.0.0.1" }
        }
    }
}
