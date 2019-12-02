from xml.etree import ElementTree as ET

def load_image():
    #データを読み込んで、パーツの位置の座標を辞書型で格納する
    
    #XMLファイルを解析
    tree = ET.parse('dataset.xml')
    root = tree.getroot()
    images = root.find("images")

    for image in list(images):
        img = ImageOps.invert(Image.open(image.get("file")).convert('L'))
        for box in list(image):
            #パーツの座標を格納する辞書
            parts = {}
            for part in list(box):
                parts[part.get("name")] = [int(part.get("x")), int(part.get("y")), 1]
                #あとで便利なので1を追加しておく
    


    #boxの左上の座標を取得する
    #offset = tree.findall('image file')
    return list(images)

print(load_image())