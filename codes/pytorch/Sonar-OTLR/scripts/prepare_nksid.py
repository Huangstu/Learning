from pathlib import Path
from zipfile import ZipFile

source = Path('external/NKSID/NKSID')
root = Path('data/raw/NKSID').resolve()
for archive in sorted(source.rglob('*.zip')):
    with ZipFile(archive) as package:
        for name in package.namelist():
            if Path(name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            label = 'tire' if archive.parent.name == 'tire' else archive.stem
            target = root / label / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = package.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Conflicting archive member: {target}')
            target.write_bytes(data)
print({p.name: len(list(p.iterdir())) for p in sorted(root.iterdir())})
