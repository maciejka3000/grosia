import httpx

class BackendService:
    def __init__(self, base_url= 'http://localhost:8000'):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
    
    async def crop_image(self, file_bytes):
        files = 