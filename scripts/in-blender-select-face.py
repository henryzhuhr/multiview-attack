import os, datetime
import bpy
import bmesh

FACES_FILE = os.path.expandvars('$HOME/BlenderModeling/aircraft_carrier/selected_faces-coverdeck.txt')
selected_face_list = []

if os.path.exists(FACES_FILE):
    with open(FACES_FILE, 'r') as f:
        for line in f.readlines():
            index = int(line.strip())
            selected_face_list.append(index)

# 加载面 上一行要空行
obj = bpy.context.object
dmesh = obj.data
bpy.ops.object.mode_set(mode='EDIT')


for face in bmesh.from_edit_mesh(obj.data).faces:
    if face.index in selected_face_list:
        face.select=True

bmesh.update_edit_mesh(dmesh) # 上一行要空行

dmesh.update() # 显示面

# 在界面选择 faces ，完成后运行
for face in bmesh.from_edit_mesh(obj.data).faces:
    if face.select and (face.index not in selected_face_list):
        selected_face_list.append(face.index)

selected_face_list

bm = bmesh.update_edit_mesh(obj.data)# 上一行要空行

SAVE_FILE=os.path.splitext(FACES_FILE)[0] + f"-{datetime.datetime.now().strftime('%m%d_%H%M')}.txt"
with open(SAVE_FILE, 'w') as f:
    for face in selected_face_list:
        f.write("%d\n" % face)
