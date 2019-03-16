import matplotlib.pyplot as plt
import re



with open('logs_baseline.txt', 'r') as f:
    data = f.read()

train_loss = re.findall(r" loss: (0\.\d+)", data)
val_loss = re.findall(r" val_loss: (0\.\d+)", data)

train_loss = list(map(lambda x : float(x), train_loss))[:48]
val_loss = list(map(lambda x : float(x), val_loss))[:48]


# with open('logs_1_nolr.txt', 'r') as f:
#     data = f.read()
#
# train_loss_nolr = re.findall(r" loss: (0\.\d+)", data)
# val_loss_nolr = re.findall(r" val_loss: (0\.\d+)", data)
#
# train_loss_nolr = list(map(lambda x : float(x), train_loss_nolr))
# val_loss_nolr = list(map(lambda x : float(x), val_loss_nolr))

with open('logs_vgg.txt', 'r') as f:
    data = f.read()

train_loss_vgg = re.findall(r" loss: (0\.\d+)", data)
val_loss_vgg = re.findall(r" val_loss: (0\.\d+)", data)

train_loss_vgg = list(map(lambda x : float(x), train_loss_vgg))[:48]
val_loss_vgg = list(map(lambda x : float(x), val_loss_vgg))[:48]

plt.plot(train_loss, "r--", color="r", label='Train Loss')
plt.plot(val_loss, "r--", color="b", label='Val Loss')
plt.plot(train_loss_vgg, color="g", label='Train Loss pretrained vgg')
plt.plot(val_loss_vgg, color="m", label='Val Loss pretrained vgg')
# plt.plot(train_loss_nolr, color="g", label='Train Loss fixed lr')
# plt.plot(val_loss_nolr, color="m", label='Val Loss fixed lr')
plt.title("Log Loss")
plt.legend()
plt.show()