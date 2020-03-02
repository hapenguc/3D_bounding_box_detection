import cv2
import numpy as np
con_path = "../../data/kitti/ImageSets_subcnn/val.txt"
val_path = "../../data/kitti/training/label_2/"
img_path = "../../exp/ddd/test/"

def project_3d_to_bird(pt):
   pt[0] += world_size / 2
   pt[1] = world_size - pt[1]
   pt = pt * out_size / world_size
   return pt.astype(np.int32)

id = 2
world_size = 64
out_size = 384
for line in open(con_path,"r"):
  number = line[:].split("\n")[0]
  #print(number)
  img_name = str(id) + "bird_pred.png"
  img_file = img_path + img_name
  print(img_file)
  bird_view = cv2.imread(img_file)
  #print(bird_view)
  with open(val_path + number +".txt") as rdf:
    for line in rdf.readlines():
      line = line.split(" ")
      if (line[0] == "Car") or (line[0] == "Pedestrian" ) or (line[0] == "Cyclist"):
        alpha = line[3]
        height, width, length = float(line[8]), float(line[9]), float(line[10])
        loca_x, loca_y, loca_z = float(line[11]), float(line[12]), float(line[13])
        rot_y = line[14]
        rot_y = float(rot_y)
        #print(rot_y)
        location = np.array([loca_x, loca_y, loca_z])
        c, s = np.cos(rot_y), np.sin(rot_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        x_corners = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        y_corners = [0,0,0,0,-height,-height,-height,-height]
        z_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners) 
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
        corners_3d = corners_3d.transpose(1, 0)
        rect = corners_3d[:4,[0,2]]
        for k in range(4):
            rect[k] = project_3d_to_bird(rect[k])
            print(rect[k])
        cv2.polylines(
          bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
          True,(100,100,100),2,lineType=cv2.LINE_AA)
        #cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
        #  (rect[e[1]][0], rect[e[1]][1]), (255,0,255), t,
        #  lineType=cv2.LINE_AA)
        for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                   (rect[e[1]][0], rect[e[1]][1]), (255,0,255), t,
                   lineType=cv2.LINE_AA)
        #print(corners_3d.shape)
        
      else:
        continue
  cv2.imwrite(img_file,bird_view)
  id += 2


