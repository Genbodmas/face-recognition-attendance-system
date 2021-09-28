import glob
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

def update_facelist():
    IMG_PATH = 'C:/Users/Gen Bodmas/PycharmProjects/CIFAR10/test_images'
    DATA_PATH = 'C:/Users/Gen Bodmas/PycharmProjects/CIFAR10'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    def trans(img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        return transform(img)


    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)

    model.eval()

    embeddings = []
    names = []
    id = []

    for usr in os.listdir(IMG_PATH):
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr) + '/*.jpg'):
            # print(usr)
            try:
                img = Image.open(file)
            except:
                continue
            with torch.no_grad():
                embeds.append(model(trans(img).to(device).unsqueeze(0)))
        if len(embeds) == 0:
            continue
        embedding = torch.cat(embeds).mean(0, keepdim=True)
        embeddings.append(embedding)
        # print(embedding)
        names.append(usr)

    embeddings = torch.cat(embeddings)
    names = np.array(names)

    torch.save(embeddings, DATA_PATH + "/faceslist.pth")
    np.save(DATA_PATH + "/usernames", names)
    print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))

    print(names)

