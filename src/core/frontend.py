import asyncio
import base64
import time
from nicegui import ui, events
import httpx


class ThemedButton(ui.button):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('color', '#f2cda0')
        super().__init__(*args, **kwargs)
    
class GrosiaApp:
    def __init__(self) -> None:
        # shared config / state
        self.backend_url = 'http://localhost:8000'
        self.http_client = httpx.AsyncClient()
        
        self.color_header = "#f2²cda0"
        self.style_button_white = 'background-color: #ffffff; cursor: pointer'
        self.style_button_cream = 'background-color: #f2cda0; cursor: pointer'
        
        
        # register pages
        ui.page('/')(self.dashboard_page)
        ui.page('/upload')(self.upload_page)
        ui.page('/receipts')(self.receipts_page)
        ui.page('/add-expense')(self.add_expense_page)

    # ---------- layout helpers ----------

    def add_header(self, cur_site: str = '/') -> None:
        def mklink(text: str, href: str) -> None:
            selected = (href == cur_site)
            bg_color = '#f2cda0' if selected else '#ffffff'

            with ui.card() \
                    .classes('border border-gray-40 p-2').style(f'background-color: {bg_color}; cursor: pointer;').props('dense'):
                ui.link(text, href).classes('text-xl font-bold text-black no-underline hover:underline')

        with ui.header(elevated=True) \
                .style('background-color: #fcba03').classes('items-center justify-between').props('bordered'):

            with ui.row().classes('items-center gap-3'):
                ui.label('Grosia').style('font-size: 24px').classes('text-xl font-bold text-black')

            with ui.row().classes('items-center flex-wrap gap-2'):
                mklink('Upload', '/upload')
                mklink('Dashboard', '/')
                mklink('Receipts', '/receipts')
                mklink('Add expense', '/add-expense')

    # ---------- pages ----------
        

    def upload_page(self):
        self.add_header('/upload')
        UploadPage(app=self)

            
    def dashboard_page(self):
        self.add_header('/')
        ui.label('Dashboard')

    def receipts_page(self):
        self.add_header('/receipts')
        ui.label('Receipts')

    def add_expense_page(self):
        self.add_header('/add-expense')
        ui.label('Add expense')

class UploadPage():
    def __init__(self, app) -> None:
        self.app = app
        
    
        with ui.card(align_items='center').classes('m-4 p-4'):
            ui.label('Upload image').classes('text-lg font-bold mb-2')
            self.uploader = ui.upload(on_upload=self.upload_page_image_handler,
                      multiple=False, auto_upload=True).props('accept="image/*,.pdf"').classes('hidden')
            
            self.button_upload = ThemedButton('Upload image', on_click=lambda: self.uploader.run_method('pickFiles'))
            self.image = ui.image('').props('width="300px" height="auto"')
            self.image.set_visibility(False)
            self.label_add_image = ui.label('No image uploaded yet.').classes('mt-2').props('width="300px"')
            with ui.row():
                self.button_rotate_left = ThemedButton('Rotate Left', on_click=self.upload_page_rotate_image)
                self.button_rotate_left.set_visibility(False)
                
                self.button_rotate_right = ThemedButton('Rotate Right', on_click = self.upload_page_rotate_image)
                self.button_rotate_right.set_visibility(False)
            
            self.button_extract_data = ThemedButton('Extract data').props('width="100%"')
            self.button_extract_data.set_visibility(False)
            
            
    async def upload_page_rotate_image(self, e:events.ClickEventArguments):
        rotate = None
        if e.sender is self.button_rotate_left:
            rotate = 'left'
        elif e.sender is self.button_rotate_right:
            rotate = 'right'
        else:
            raise ValueError('wrong button')
        print(rotate)
        
        
            
    async def upload_page_image_handler(self, e : events.UploadEventArguments):
        # e.files is a list of UploadedFile
        self.label_add_image.set_visibility(True)
        self.label_add_image.set_text('Processing uploaded image...')
        uploaded_file = e.file
        print('name:', uploaded_file.name)
        print('content_type:', uploaded_file.content_type)
        print('path:', uploaded_file._path)
        data = uploaded_file._path.read_bytes()
        
        # build data URL for preview
        mime = uploaded_file.content_type or 'image/jpeg'
        b64 = base64.b64encode(data).decode('ascii')
        self.image.source = f'data:{mime};base64,{b64}'

        # toggle placeholder vs image
        self.label_add_image.set_visibility(False)
        self.image.set_visibility(True)
        self.label_add_image.set_text('')  # optional
        
        self.label_add_image.set_visibility(False)
        uploaded_file = e.file
        
        self.button_rotate_left.set_visibility(True)
        self.button_rotate_right.set_visibility(True)
        self.button_extract_data.set_visibility(True)
        

app = GrosiaApp()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(reload=True)