import json
import cv2
import numpy as np
#%%
#with open('./tmp_lines_record.json') as f:
#    record_lines = json.load(f)

#for img in record_lines:
#    imgname = img['imgname']
#    depth_file = imgname.replace('rgb','sync_depth').replace('jpg','png')
#    print(depth_file)
# %%
#depth = cv2.imread("/home/pebert/dataset/nyu_depth_v2/official_splits/test/study_room/sync_depth_00272.png", -1)
#depth = depth.astype(np.float32) / 1000.0

#print(np.array(depth)[100,:])
#cv2.imwrite('home/pebert/bts/tmp.jpg',depth/10*255)
# %%
pred_depth = cv2.imread('/home/pebert/bts/pytorch/result_wireframe/raw/00030043_0_depth.jpg',-1)
pred_depth = pred_depth.astype(np.float32) / 1000.0

pred_depth[pred_depth < 1e-3] = 1e-3
pred_depth[pred_depth > 10] = 10
pred_depth[np.isinf(pred_depth)] = 10
pred_depth[np.isnan(pred_depth)] = 1e-3
cv2.imwrite('/home/pebert/bts/tmp_pred.jpg',pred_depth/10*255)