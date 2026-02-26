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
- Ecrit les images recadrees dans `Pictures/Cropped`
- Charge le modele dans `Models/` (par defaut: `Models/yolov8n.pt`)
- Garde le plus gros chat detecte par image
- Deplace les photos sources:
  - `Pictures/Ok` si un chat est detecte
  - `Pictures/Skip` sinon
- Ne vide pas `Pictures/Cropped` automatiquement

## Options utiles

```powershell
python crop_cat.py --padding 0.20
python crop_cat.py --output-box
python crop_cat.py --conf 0.40
python crop_cat.py --model Models/yolo12n.pt
python crop_cat.py --retry all
python crop_cat.py --retry skip
python crop_cat.py --retry ok
python crop_cat.py --input "D:\MesPhotos\Chats" --output "D:\MesPhotos\Chats\Cropped"
```

`--retry all` vide `Cropped`, remet les images de `Ok` et `Skip` dans `Pictures`, puis relance tout.
`--retry skip` remet seulement les images de `Skip` dans `Pictures`, puis relance.
`--retry ok` remet seulement les images de `Ok` dans `Pictures`, puis relance.
`--output-box` force un recadrage carre qui englobe la box chat detectee (avec padding) si possible; sinon la box est reduite juste assez pour rentrer, en gardant la tete dans le cadre si possible.
Pour le focus tete via cascade OpenCV, installer en plus:

```powershell
python -m pip install opencv-python
```

## Logs

- `[ok]`: image traitee
- `[skip]`: aucun chat detecte
- `[done] Saved: X | Skipped: Y | Output: ...`
