import os
import bpy 
import os
import sys
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
# remove all ', , and [ ] from the args
argv = [x.replace("'", "") for x in argv]
argv = [x.replace(",", "") for x in argv]
argv = [x.replace("[", "") for x in argv]
argv = [x.replace("]", "") for x in argv]
output_path = argv[1]
envmap_path = argv[0]

# open blender file
bpy.ops.wm.open_mainfile(filepath="assets/scene_9_spheres.blend")

scene = bpy.context.scene

scene.render.image_settings.file_format='OPEN_EXR'

scene.render.filepath= 'ambient_parametric'
bpy.data.images[0].filepath = envmap_path

bpy.data.objects["Sphere.001"].hide_render = True

scene.render.filepath=output_path

bpy.ops.render.render(write_still=True)