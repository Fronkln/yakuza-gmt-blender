import importlib.util

blender_loader = importlib.util.find_spec('bpy')

# Include the bl_info at the top level always
bl_info = {
    "name": "Yakuza GMT File Import/Export",
    "author": "SutandoTsukai181",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "description": "Import-Export Yakuza GMT Files",
    "warning": "",
    "doc_url": "https://github.com/SutandoTsukai181/yakuza-gmt-blender/wiki",
    "tracker_url": "https://github.com/SutandoTsukai181/yakuza-gmt-blender/issues",
    "category": "Import-Export",
}

if blender_loader:
    from .blender.addon import *


from . import addon_updater_ops
from .addon_updater_prefs import GMTUpdaterPreferences


classes = (
    GMTUpdaterPreferences,
)


def register():
    addon_updater_ops.register(bl_info)

    for c in classes:
        bpy.utils.register_class(c)

    # Check for update as soon as the updater is loaded
    # If auto check is enabled and the conditions are met, will
    # display a pop up once the user clicks anywhere in the scene
    addon_updater_ops.check_for_update_background()

    register_addon()


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    addon_updater_ops.unregister()

    unregister_addon()
