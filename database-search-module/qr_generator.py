import qrcode

dog_id = "BLR-6001"

qr = qrcode.make(dog_id)

filename = f"{dog_id}_QR.png"
qr.save(filename)

print(f"QR code generated successfully: {filename}")