import asyncio
from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Bot

import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

queue = asyncio.Queue()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправьте изображение как файл, и я увеличу его разрешение."
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document:
        await update.message.reply_text("Пожалуйста, отправьте именно файл изображения.")
        return

    file = await document.get_file()
    file_bytes = BytesIO()
    await file.download_to_memory(out=file_bytes)
    file_bytes.seek(0)

    await queue.put((update.effective_chat.id, file_bytes, document.file_name))
    await update.message.reply_text("Файл принят. Начинаем обработку...")

async def worker(app: Application):
    while True:
        chat_id, file_bytes, file_name_raw = await queue.get()

        img = Image.open(file_bytes)

        new_size = (img.width * 2, img.height * 2)
        upscaled = img.resize(new_size, Image.BICUBIC)

        out_bytes = BytesIO()
        upscaled.save(out_bytes, format="PNG")
        out_bytes.seek(0)

        bot : Bot = app.bot
        file_name, file_ext = file_name_raw.rsplit('.', 1)
        await bot.send_document(chat_id, document=out_bytes, filename=f'{file_name}.x2.{file_ext}')

        queue.task_done()

def main():
    async def post_init(app):
        app.create_task(worker(app))

    app : Application = ( ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_file))

    app.run_polling()

if __name__ == "__main__":
    main()
