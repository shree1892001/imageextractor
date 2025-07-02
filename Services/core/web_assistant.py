import asyncio
from typing import Optional
from playwright.async_api import async_playwright
from elements import ElementContext, ElementType
from dropdown import DropdownContext, DropdownType
from Common.constants import API_KEY

class WebAssistant:
    """Main web assistant class that coordinates UI interactions"""
    
    def __init__(self, llm_selector, page, voice_engine):
        self.llm_selector = llm_selector
        self.page = page
        self.voice_engine = voice_engine
        self.element_handlers = self._initialize_element_handlers()
        self.is_running = False

    def _initialize_element_handlers(self):
        """Initialize handlers for different element types"""
        from .dropdown import Dropdown
        # Import other element handlers here
        
        return {
            ElementType.DROPDOWN: Dropdown(self.page, self.llm_selector, self.voice_engine),
            # Add other element handlers here
        }

    async def _analyze_command(self, command: str) -> Optional[ElementContext]:
        """Analyze voice command and create appropriate context"""
        analysis = await self._get_command_analysis(command)
        if not analysis:
            return None

        page_info = self._get_page_context()
        
        if analysis["element_type"] == "dropdown":
            return DropdownContext(
                purpose=analysis["purpose"],
                surrounding_text=" ".join(analysis["keywords"]),
                element_type=ElementType.DROPDOWN,
                page_content=page_info.get("visibleText", ""),
                url=page_info.get("url", ""),
                title=page_info.get("title", ""),
                dropdown_type=DropdownType[analysis["dropdown_type"].upper()]
            )
        
        return ElementContext(
            purpose=analysis["purpose"],
            surrounding_text=" ".join(analysis["keywords"]),
            element_type=ElementType[analysis["element_type"].upper()],
            page_content=page_info.get("visibleText", ""),
            url=page_info.get("url", ""),
            title=page_info.get("title", "")
        )

    async def handle_command(self, command: str) -> bool:
        """Main command handling method"""
        context = await self._analyze_command(command)
        if not context:
            return False

        handler = self.element_handlers.get(context.element_type)
        if not handler:
            return False

        return await handler.interact(context)

    async def start(self, initial_url: str = "https://www.google.com"):
        """Start the web assistant with an initial URL"""
        self.is_running = True
        try:
            await self.page.goto(initial_url)
            print(f"🌐 Navigated to {initial_url}")
            print("👂 Listening for commands... (say 'exit' to quit)")
            
            while self.is_running:
                command = await self.voice_engine.listen()
                if not command:
                    continue

                if command.lower() in ["exit", "quit", "stop"]:
                    self.is_running = False
                    print("👋 Goodbye!")
                    continue

                print(f"🎤 Command received: {command}")
                success = await self.handle_command(command)
                if not success:
                    print("❌ Command failed or not recognized")
                else:
                    print("✅ Command executed successfully")

        except Exception as e:
            print(f"❌ Error: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the web assistant and cleanup resources"""
        self.is_running = False
        try:
            await self.page.close()
            print("🛑 Browser closed")
        except Exception as e:
            print(f"Error closing browser: {e}")

async def main():
    """Main function to run the web assistant"""
    try:
        # Initialize required components
        from your_llm_module import LLMSelector  # Replace with your actual LLM module
        from your_voice_module import VoiceEngine  # Replace with your actual voice module
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            # Initialize components
            llm_selector = LLMSelector(API_KEY)
            voice_engine = VoiceEngine()
            
            # Create and start web assistant
            assistant = WebAssistant(llm_selector, page, voice_engine)
            
            print("""
╔════════════════════════════════════════════════════════╗
║                   WEB ASSISTANT                         ║
║ ---------------------------------------------------- ║
║ Commands:                                              ║
║  - "click [element]"                                   ║
║  - "type [text] in [field]"                           ║
║  - "select [option] from [dropdown]"                   ║
║  - "exit" to quit                                      ║
╚════════════════════════════════════════════════════════╝
            """)
            
            try:
                await assistant.start()
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
            finally:
                await assistant.stop()
                await browser.close()
    
    except Exception as e:
        print(f"Critical error: {e}")
        print("Please check browser drivers and dependencies.")

if __name__ == "__main__":
    asyncio.run(main())