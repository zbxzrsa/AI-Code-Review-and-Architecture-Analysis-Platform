"""
WebSocket API端点
提供实时状态推送和事件订阅
"""

import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import HTMLResponse

from app.services.websocket_service import (
    connection_manager, 
    event_service,
    handle_websocket_connection,
    EventType
)
from app.core.auth import get_current_user_optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/events")
async def websocket_events_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    pr_number: Optional[int] = Query(None),
    tenant_id: Optional[str] = Query(None),
    repo_id: Optional[str] = Query(None),
    subscriptions: Optional[str] = Query(None)  # 逗号分隔的订阅列表
):
    """WebSocket事件端点"""
    
    # 解析订阅列表
    subscription_list = []
    if subscriptions:
        subscription_list = [sub.strip() for sub in subscriptions.split(",")]
    
    # 默认订阅
    if not subscription_list:
        subscription_list = ["global"]
    
    # 添加特定订阅
    if session_id:
        subscription_list.append(f"session:{session_id}")
    if pr_number:
        subscription_list.append(f"pr:{pr_number}")
    if tenant_id:
        subscription_list.append(f"tenant:{tenant_id}")
    if repo_id:
        subscription_list.append(f"repo:{repo_id}")
    
    # 连接元数据
    metadata = {
        "session_id": session_id,
        "pr_number": pr_number,
        "tenant_id": tenant_id,
        "repo_id": repo_id,
        "user_agent": websocket.headers.get("user-agent"),
        "client_ip": websocket.client.host if websocket.client else None
    }
    
    await handle_websocket_connection(
        websocket=websocket,
        connection_id=session_id,  # 使用session_id作为连接ID
        subscriptions=subscription_list,
        metadata=metadata
    )


@router.get("/events/history")
async def get_event_history(
    session_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: Dict = Depends(get_current_user_optional)
):
    """获取事件历史"""
    
    # 验证事件类型
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            return {"error": f"Invalid event type: {event_type}"}
    
    history = event_service.get_event_history(
        session_id=session_id,
        event_type=event_type_enum,
        limit=limit
    )
    
    return {
        "history": history,
        "total_count": len(history),
        "filters": {
            "session_id": session_id,
            "event_type": event_type,
            "limit": limit
        }
    }


@router.get("/events/stats")
async def get_event_stats(
    current_user: Dict = Depends(get_current_user_optional)
):
    """获取事件统计"""
    return event_service.get_event_stats()


@router.get("/connections")
async def get_connections_info(
    current_user: Dict = Depends(get_current_user_optional)
):
    """获取连接信息"""
    return {
        "active_connections": connection_manager.get_connection_count(),
        "subscriptions": {
            key: connection_manager.get_subscription_count(key)
            for key in connection_manager.subscriptions.keys()
        }
    }


@router.get("/test")
async def websocket_test_page():
    """WebSocket测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .controls { margin-bottom: 20px; }
            .messages { 
                border: 1px solid #ccc; 
                height: 400px; 
                overflow-y: auto; 
                padding: 10px;
                background-color: #f9f9f9;
            }
            .message { 
                margin-bottom: 5px; 
                padding: 5px;
                border-radius: 3px;
            }
            .analysis-progress { background-color: #e3f2fd; }
            .analysis-completed { background-color: #e8f5e8; }
            .analysis-failed { background-color: #ffebee; }
            .system-status { background-color: #fff3e0; }
            .timestamp { color: #666; font-size: 0.8em; }
            .event-type { font-weight: bold; }
            input, button, select { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WebSocket Real-time Events Test</h1>
            
            <div class="controls">
                <div>
                    <label>Session ID:</label>
                    <input type="text" id="sessionId" value="test-session-123">
                </div>
                <div>
                    <label>PR Number:</label>
                    <input type="number" id="prNumber" value="123">
                </div>
                <div>
                    <label>Tenant ID:</label>
                    <input type="text" id="tenantId" value="tenant-1">
                </div>
                <div>
                    <label>Repo ID:</label>
                    <input type="text" id="repoId" value="repo-1">
                </div>
                <div>
                    <label>Subscriptions:</label>
                    <input type="text" id="subscriptions" value="global,event_type:analysis_progress" style="width: 300px;">
                </div>
                <div>
                    <button onclick="connect()">Connect</button>
                    <button onclick="disconnect()">Disconnect</button>
                    <button onclick="clearMessages()">Clear Messages</button>
                </div>
            </div>
            
            <div class="messages" id="messages"></div>
            
            <div>
                <h3>Send Test Event</h3>
                <select id="eventType">
                    <option value="analysis_progress">Analysis Progress</option>
                    <option value="analysis_completed">Analysis Completed</option>
                    <option value="analysis_failed">Analysis Failed</option>
                    <option value="system_status">System Status</option>
                </select>
                <button onclick="sendTestEvent()">Send Test Event</button>
            </div>
        </div>

        <script>
            let ws = null;
            let messageCount = 0;
            
            function connect() {
                if (ws) {
                    ws.close();
                }
                
                const sessionId = document.getElementById('sessionId').value;
                const prNumber = document.getElementById('prNumber').value;
                const tenantId = document.getElementById('tenantId').value;
                const repoId = document.getElementById('repoId').value;
                const subscriptions = document.getElementById('subscriptions').value;
                
                const wsUrl = `ws://localhost:8000/ws/events?session_id=${sessionId}&pr_number=${prNumber}&tenant_id=${tenantId}&repo_id=${repoId}&subscriptions=${encodeURIComponent(subscriptions)}`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    addMessage('Connected to WebSocket', 'system-status');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage(JSON.stringify(data, null, 2), data.event_type);
                };
                
                ws.onclose = function(event) {
                    addMessage('WebSocket connection closed', 'system-status');
                };
                
                ws.onerror = function(error) {
                    addMessage('WebSocket error: ' + error, 'analysis-failed');
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function addMessage(content, eventType = 'info') {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${eventType}`;
                
                const timestamp = new Date().toLocaleTimeString();
                messageDiv.innerHTML = `
                    <div class="event-type">${eventType || 'info'}</div>
                    <div class="timestamp">${timestamp}</div>
                    <pre>${content}</pre>
                `;
                
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
                
                messageCount++;
                if (messageCount > 100) {
                    messages.removeChild(messages.firstChild);
                    messageCount--;
                }
            }
            
            function clearMessages() {
                document.getElementById('messages').innerHTML = '';
                messageCount = 0;
            }
            
            function sendTestEvent() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    addMessage('WebSocket not connected', 'analysis-failed');
                    return;
                }
                
                const eventType = document.getElementById('eventType').value;
                const sessionId = document.getElementById('sessionId').value;
                const prNumber = document.getElementById('prNumber').value;
                
                let testEvent;
                
                switch(eventType) {
                    case 'analysis_progress':
                        testEvent = {
                            type: 'subscribe',
                            channel: `session:${sessionId}`
                        };
                        break;
                    case 'analysis_completed':
                        testEvent = {
                            type: 'ping'
                        };
                        break;
                    default:
                        testEvent = {
                            type: 'ping'
                        };
                }
                
                ws.send(JSON.stringify(testEvent));
                addMessage('Sent: ' + JSON.stringify(testEvent), 'info');
            }
            
            // 自动连接
            setTimeout(connect, 100);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/events/test")
async def send_test_event(
    event_type: str,
    session_id: Optional[str] = None,
    pr_number: Optional[int] = None,
    current_user: Dict = Depends(get_current_user_optional)
):
    """发送测试事件（仅用于开发测试）"""
    
    if event_type == "analysis_progress":
        await event_service.publish_analysis_progress(
            session_id=session_id or "test-session",
            progress=0.75,
            current_file="src/example.py",
            total_files=100,
            processed_files=75,
            pr_number=pr_number,
            tenant_id="test-tenant",
            repo_id="test-repo"
        )
    elif event_type == "analysis_completed":
        await event_service.publish_analysis_completed(
            session_id=session_id or "test-session",
            pr_number=pr_number or 123,
            analysis_results={
                "total_issues": 5,
                "critical_issues": 1,
                "warning_issues": 4
            },
            performance_metrics={
                "analysis_time": 45.2,
                "cache_hit_ratio": 0.65
            },
            tenant_id="test-tenant",
            repo_id="test-repo"
        )
    elif event_type == "system_status":
        await event_service.publish_system_status({
            "cpu_usage": 0.45,
            "memory_usage": 0.67,
            "active_tasks": 3,
            "queue_size": 12
        })
    
    return {"message": f"Test event {event_type} sent successfully"}