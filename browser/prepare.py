import io,numpy as np
from PIL import Image
def prepare(raw):
 im=Image.open(io.BytesIO(raw)).convert('RGB')
 w,h=im.size
 size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
 im=im.resize(size,Image.Resampling.BILINEAR)
 w,h=im.size
 x,y=round((w-224)/2),round((h-224)/2)
 a=np.asarray(im.crop((x,y,x+224,y+224)),dtype=np.float32)/255.
 a=(a-np.asarray([.485,.456,.406],dtype=np.float32))/np.asarray([.229,.224,.225],dtype=np.float32)
 return a.transpose(2,0,1).copy().ravel()
