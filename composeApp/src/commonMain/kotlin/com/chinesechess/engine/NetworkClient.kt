package com.chinesechess.engine

data class OnlineUser(
    val name: String,
    val ip: String,
    val port: Int,
    val lastSeen: Long
)

data class InviteInfo(
    val fromName: String,
    val fromIp: String,
    val fromPort: Int,
    val fromSide: PSide
)

interface NetworkClient {
    val myIp: String
    val isOnline: Boolean
    var myOnlineSide: PSide
    val discoveredUsers: List<OnlineUser>

    fun init(userName: String)
    fun cleanup()
    fun broadcast()
    fun poll(): InviteInfo?
    fun refreshUsers()
    fun sendInvite(ip: String, port: Int, side: PSide)
    fun acceptInvite(side: PSide)
    fun recvMove(): Move?
    fun sendMove(move: Move)
}
