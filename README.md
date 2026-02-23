# Rudy - Crop automatique de chats

Script Python pour detecter les chats sur des photos et generer des images recadrees.

## Prerequis

- Python 3.10+
- PowerShell (Windows)

Verification:

```powershell
python --version
```

## Installation

Depuis le dossier du projet:

```powershell
python -m pip install -e .
```

## Lancement

Commande simple:

```powershell
python crop_cat.py
```

Ou via le script Windows:

```powershell
.\run_crop_cat.bat
```

## Comportement par defaut

- Lit les images dans `Pictures`
- Ecrit les images recadrees dans `Pictures/CROPPED`
- Charge le modele dans `Models/` (par defaut: `Models/yolov8n.pt`)
- Garde le plus gros chat detecte par image
- Deplace les photos sources:
  - `Pictures/OK` si un chat est detecte
  - `Pictures/SKIP` sinon
- Ne vide pas `Pictures/CROPPED` automatiquement

## Options utiles

```powershell
python crop_cat.py --padding 0.20
python crop_cat.py --conf 0.40
python crop_cat.py --model Models/yolo12n.pt
python crop_cat.py --retry-all
python crop_cat.py --retry-skip
python crop_cat.py --input "D:\MesPhotos\Chats" --output "D:\MesPhotos\Chats\CROPPED"
```

`--retry-all` vide `CROPPED`, remet les images de `OK` et `SKIP` dans `Pictures`, puis relance tout.
`--retry-skip` remet seulement les images de `SKIP` dans `Pictures`, puis relance.

## Logs

- `[ok]`: image traitee
- `[skip]`: aucun chat detecte
- `[done] Saved: X | Skipped: Y | Output: ...`
