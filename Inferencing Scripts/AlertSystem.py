import os, logging, smtplib,json, uuid, threading, httpx
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from cryptography.fernet import Fernet
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio,time
from telegram.request import HTTPXRequest
from telegram.error import RetryAfter


from dotenv import load_dotenv

load_dotenv("load_env.env")

logger = logging.getLogger(__name__)

KEY = os.getenv('FERNET_KEY')
if not KEY:
    logger.error("FERNET_KEY not set. Exiting...")

cipher = Fernet(KEY)
if not cipher:
    logger.error("Failed to create Fernet cipher. Exiting...", exc_info=True)
    raise

class Telegram:
    def __init__(self):
        self.TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
        self.BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME")
        self.USERS_FILE = "users.json"
        self.TOKENS_FILE = "tokens.json"
        self.last_alert = None

        # ─── create one shared HTTPXRequest with a big connection pool ───
        self.request = HTTPXRequest(
            connection_pool_size=50,
            pool_timeout=60.0,
            connect_timeout=20.0,
            read_timeout=300.0,
            write_timeout=120.0
        )
        # pass it into both Bot and later into Application
        self.bot = Bot(token=self.TELEGRAM_TOKEN, request=self.request)
        self.app = None
        self._polling_started = False

        if not self.TELEGRAM_TOKEN or not self.BOT_USERNAME:
            logger.error("Telegram config incomplete", exc_info=True)
            raise RuntimeError("Missing Telegram credentials")

    # ─── JSON Helpers ─────────────────────────────
    def load_json(self, path):
        try:
            return json.load(open(path))
        except:
            return {}

    def save_json(self, path, data):
        json.dump(data, open(path, "w"), indent=4)

    # ─── Token Management ─────────────────────────
    def generate_token(self, internal_id):
        tokens = self.load_json(self.TOKENS_FILE)
        t = uuid.uuid4().hex
        tokens[t] = internal_id
        self.save_json(self.TOKENS_FILE, tokens)
        return t

    def pop_token(self, t):
        tokens = self.load_json(self.TOKENS_FILE)
        user = tokens.pop(t, None)
        if user:
            self.save_json(self.TOKENS_FILE, tokens)
        return user

    # ─── Subscription Management ───────────────────
    def set_user(self, chat_id, internal_id, status):
        users = self.load_json(self.USERS_FILE)
        users[encryptor(str(chat_id))] = {
            "internal": encryptor(internal_id),
            "subscribed": status
        }
        self.save_json(self.USERS_FILE, users)
        logger.info(f"User {chat_id} subscribed: {status}", exc_info=True)
      
    def get_subscribed_users(self):
        users = self.load_json(self.USERS_FILE)
        return [
            int(decryptor(cid))
            for cid, data in users.items()
            if data.get("subscribed", False)
        ]

    # ─── Telegram Commands ─────────────────────────
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        token = context.args[0] if context.args else None

        if not token:
            await update.message.reply_text("❌ Invalid start command.")
            return

        internal_id = self.pop_token(token)
        if not internal_id:
            await update.message.reply_text("❌ Invalid or expired token.")
            return

        self.set_user(chat_id, internal_id, True)
        await update.message.reply_text("✅ Notifications enabled. You will now receive anomaly alerts. Send /stop to unsubscribe.")

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        users = self.load_json(self.USERS_FILE)

        # Search for the encrypted chat_id
        encrypted_chat_id = None
        for enc_cid in users.keys():
            try:
                if decryptor(enc_cid) == str(chat_id):
                    encrypted_chat_id = enc_cid
                    break
            except Exception as e:
                logger.warning(f"Decryption failed for key: {enc_cid} with error: {e}")
                continue

        if encrypted_chat_id:
            user_data = users[encrypted_chat_id]
        
            internal = decryptor(user_data.get("internal"))
            self.set_user(chat_id, internal, False)
            await update.message.reply_text("❌ Notifications disabled. You will no longer receive alerts.")
        else:
            await update.message.reply_text("⚠️ You were not subscribed.")


    async def send_anomaly_alert(self, msg: str):
        """ Send to each subscribed chat, with flood-control handling. """
        for cid in self.get_subscribed_users():
            for attempt in range(3):
                try:
                    await self.bot.send_message(chat_id=cid, text=msg)
                    # throttle ≤1 msg/sec per chat
                    await asyncio.sleep(1)
                    break
                except RetryAfter as e:
                    logger.warning(f"Flood limit hit for {cid}, sleeping {e.retry_after}s")
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    logger.warning(f"Send to {cid} failed on attempt {attempt+1}: {e}")
                    await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"All retries failed for {cid}")


    def get_link(self, user_internal_id):
        token = self.generate_token(user_internal_id)
        return f"[🔗Click here to subscribe to Telegram](https://t.me/{self.BOT_USERNAME}?start={token})"


    def start_bot(self):
        if self._polling_started:
            return
        self._polling_started = True

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop                 # ← save it here

            loop.run_until_complete(
                self.bot.delete_webhook(drop_pending_updates=True)
            )
            app = (
                Application.builder()
                        .token(self.TELEGRAM_TOKEN)
                        .request(self.request)
                        .build()
            )

            app.add_handler(CommandHandler("start", self.start))
            app.add_handler(CommandHandler("stop",  self.stop))

            self.app = app
          
            app.run_polling(
                drop_pending_updates=True,  
                timeout=20,                 
                poll_interval=0, close_loop=False)

        threading.Thread(target=_run, daemon=True).start()

    def send_telegram(self, message: str):
        if message == self.last_alert:
            return
        self.last_alert = message  
        
        if not getattr(self, "loop", None):
            logger.error("Bot loop not running—call start_bot() first.")
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self.send_anomaly_alert(message),
                self.loop
            )
        except httpx.ConnectError as e:
            logger.error(f"Telegram connection error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error while sending Telegram message: {e}", exc_info=True)
        logger.info(f"Telegram message sent: {message}")

class Email:
    def __init__(self):
        self.address = os.getenv('EMAIL_ADDRESS')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.smtp_host = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))

        if not all([self.address, self.password, self.smtp_host]):
            logger.error("Email configuration incomplete; set EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER.")
            raise RuntimeError("Incomplete email configuration")

        logger.debug(f"Email configured for {self.address}@{self.smtp_host}:{self.smtp_port}")

    def send_email(self, recipient: str, body: str, max_retries: int = 3):
        msg = MIMEMultipart()
        msg['From'] = self.address
        msg['To'] = recipient
        msg['Subject'] = "Anomaly detection Alert"
        msg.attach(MIMEText(body, 'plain'))

        delay = 1
        for attempt in range(1, max_retries + 1):
            try:
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port,timeout=10) as server:
                    server.login(self.address, self.password)
                    server.sendmail(self.address, recipient, msg.as_string())
                logger.info(f"Email sent successfully to {recipient}")
                break
            except smtplib.SMTPConnectError as e:
                logger.warning(f"[Attempt {attempt}] SMTP connect error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                logger.exception(f"Failed to send email on attempt {attempt}: {e}")
                raise

def encryptor(text):
    """
    Encrypt the text using Fernet (AES) and return a base64-encoded string.
    """
    encrypted = cipher.encrypt(str(text).encode())
    return encrypted.decode()  

def decryptor(encrypted_text):
    """
    Decrypt the text using Fernet (AES) from a base64-encoded string.
    """
    decrypted = cipher.decrypt(encrypted_text.encode())  
    return decrypted.decode()


