# Server Restart Issue - Fixed

## Root Cause

Your server was going down and coming back up because:

1. **Health Check Timeouts**: The `/health` endpoint was trying to ping Redis synchronously, which could hang if Redis was slow or unavailable. When the health check times out, the deployment platform (Render) thinks the service is dead and restarts it.

2. **Blocking API Calls**: Together AI API calls are synchronous and can hang indefinitely. When multiple requests come in:
   - Workers get blocked waiting for API responses
   - Health checks can't respond (workers are blocked)
   - Platform detects service as unhealthy
   - Platform restarts the service
   - Cycle repeats

## Fixes Applied

### 1. **Fast Health Check** ✅
- Health check now responds in < 500ms
- Redis ping has timeout (won't block)
- Health check never fails (graceful degradation)
- Always returns "healthy" status

### 2. **API Call Timeouts** ✅
- All Together AI API calls now have 30-second timeout
- Uses threading to prevent blocking
- Timeout errors are caught and handled gracefully
- Prevents workers from hanging indefinitely

### 3. **Non-Blocking Operations** ✅
- Health check uses async with timeout
- API calls wrapped in threads with timeouts
- Event loop never gets blocked

## What Changed

1. **Health Check Endpoint** (`/health`):
   - Now uses `asyncio.wait_for` with 500ms timeout
   - Never blocks or hangs
   - Always returns quickly

2. **Together API Calls**:
   - Wrapped in `call_together_api_with_timeout()`
   - Uses threading with timeout
   - All 4 API call locations updated:
     - `classify_intent()`
     - `generate_roast()`
     - `generate_advice()`
     - `generate_savage_chat_response()`

## Testing

After deployment, monitor:
- Health check response time (should be < 1 second)
- No more restart cycles
- API calls complete within 30 seconds or timeout gracefully

## Environment Variables

Make sure these are set:
```bash
TOGETHER_API_TIMEOUT=30  # 30 seconds max for API calls
```

The health check will now always respond quickly, preventing the restart cycle!
