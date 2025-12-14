# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Pipe Pile Optimizer
Creates a standalone Mac .app bundle
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the pulp solver directory
import pulp
pulp_path = os.path.dirname(pulp.__file__)
solver_dir = os.path.join(pulp_path, 'solverdir')

block_cipher = None

# Exclude heavy unused packages
EXCLUDES = [
    # ML/DL frameworks (not needed)
    'torch', 'tensorflow', 'keras', 'transformers',
    'sklearn', 'scikit-learn',
    # Visualization (not needed - we use tkinter)
    'matplotlib', 'seaborn', 'plotly', 'bokeh',
    # Jupyter/IPython
    'IPython', 'jupyter', 'notebook', 'ipykernel',
    # Heavy optional dependencies
    'scipy.spatial.cKDTree',
    'PIL', 'cv2', 'imageio',
    # Testing
    'pytest', 'hypothesis',
    # Other unused
    'sqlalchemy', 'alembic',
    'boto3', 'botocore', 's3transfer',
    'google', 'grpc', 'opentelemetry',
    'cryptography', 'nacl',
    'lxml', 'bs4', 'html5lib',
    'jedi', 'parso',
]

a = Analysis(
    ['pipe_optimizer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include the optimizer module
        ('pipe_optimizer_v5_safe.py', '.'),
        # Include PuLP solvers (CBC)
        (solver_dir, 'pulp/solverdir'),
    ],
    hiddenimports=[
        'pipe_optimizer_v5_safe',
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs.base',
        'numpy',
        'openpyxl',
        'pulp',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PipeOptimizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=True,  # Important for Mac .app
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PipeOptimizer',
)

app = BUNDLE(
    coll,
    name='Pipe Optimizer.app',
    icon=None,
    bundle_identifier='com.pipes.optimizer',
    info_plist={
        'CFBundleName': 'Pipe Optimizer',
        'CFBundleDisplayName': 'Pipe Pile Optimizer',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'NSRequiresAquaSystemAppearance': 'False',  # Support dark mode
    },
)
