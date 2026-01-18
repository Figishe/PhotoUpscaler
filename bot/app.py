import asyncio
from io import BytesIO
from PIL import Image

import telegram
from telegram import Update, Message, Document
from telegram.ext import ApplicationBuilder, Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Bot

from model.lit_upscaler import LitSuperResNet
from model.inference import Inference

import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

queue = asyncio.Queue()

inference: Inference

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assert update.message, "message can't be null in start handler"
    await update.message.reply_text(
        "Привет! Отправьте изображение как файл, и я увеличу его разрешение."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assert update.message, "message can't be null in message handler"

    if update.message.photo:
        await update.message.reply_text("Пожалуйста, отправьте изображение именно как файл")
        return

    document = update.message.document
    if not document:
        await update.message.reply_text("Пожалуйста, отправьте файл изображения")
        return
    
    if document.mime_type is None or not document.mime_type.startswith("image/"):
        await update.message.reply_text(f"Незнакомый тип файла {document.mime_type}; поддерживаются jped, png и др.")
        return

    try:
        file = await document.get_file()
    except telegram.error.TelegramError as e:
        if "File is too big" in str(e):
            await update.message.reply_text(f"😢 Извините, Telegram не даёт мне скачать такой большой файл.\n"
                    "Попробуйте отправить файл весом поменьше, например, в поджатом jpeg."
                )
        
        return
    
    file_bytes = BytesIO()
    await file.download_to_memory(out=file_bytes)
    file_bytes.seek(0)

    await update.message.reply_text("Файл принят. Скоро скину увеличенный... ⏳")
    await queue.put((update.effective_chat.id, file_bytes, document.file_name))
    

async def worker(app: Application):
    while True:
        chat_id, file_bytes, file_name_raw = await queue.get()
        file_name, file_ext = file_name_raw.rsplit('.', 1)

        img = Image.open(file_bytes)

        upscaled = inference.upscale(img)

        out_bytes = BytesIO()
        upscaled.save(out_bytes, format=img.format, quality=95 if img.format == "JPEG" else None)
        out_bytes.seek(0)
        out_size_mb = out_bytes.getbuffer().nbytes / (1024 * 1024)
        
        try:
            await app.bot.send_document(chat_id, document=out_bytes, filename=f'{file_name}.x2.{file_ext}')
        except telegram.error.BadRequest as e:
            if "File is too big" in str(e):
                await app.bot.send_message(
                    chat_id,
                    f"😢 Извините, Telegram не даёт мне вернуть вам файл размером {out_size_mb:.1f} Mb.\n"
                    "Попробуйте отправить файл весом поменьше, например, в поджатом jpeg."
                )
            else:
                await app.bot.send_message(chat_id, f"😢 Не получилось отправить вам результат: {e}")

        queue.task_done()


def load_model():
    model = LitSuperResNet.load_from_checkpoint(
        'checkpoints/upscaler.ckpt',
    )

    global inference
    inference = Inference(model)


def main():
    if BOT_TOKEN is None:
        raise Exception("BOT_TOKEN should be specified in .env")

    load_model()

    async def post_init(app):
        app.create_task(worker(app))

    app : Application = ( ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters=None, callback=handle_message))

    app.run_polling()

if __name__ == "__main__":
    main()
