# AUR packaging

This repository is packaged for Arch Linux as two AUR packages (locally
prepared, not yet submitted):

| Package | Contents |
|---|---|
| `birdid-git` | the pipeline (`/usr/share/birdid/pipeline/`), the eBird seed data (`/usr/share/birdid/ebird/`), the `/usr/bin/birdid` wrapper, LICENSE |
| `python-pybioclip` | the only missing dependency (`pybioclip` 2.1.6 from PyPI); everything else in the chain is in official `extra` / AUR |

## Install layout

```
/usr/share/birdid/
├── pipeline/            # birdid (launcher), birdid.py, ebird_data.py, README
└── ebird/               # read-only seed data (taxonomy + curated region seeds)
/usr/bin/birdid          # wrapper -> /usr/share/birdid/pipeline/birdid
```

The launcher resolves its own location (`SCRIPT_DIR`) to find `birdid.py`, and
the pipeline resolves the data dir relative to it, so the siblings must stay
next to each other.

## Why a wrapper?

`--download` / `--update` / `--initialize` write `regions_generated/` and the
downloaded taxonomy **into the data dir**, which is read-only under
`/usr/share`. `packaging/birdid-wrapper` (installed as `/usr/bin/birdid`):

1. provisions a user-writable copy of the seed data at
   `$XDG_DATA_HOME/birdid/ebird` (or `~/.local/share/birdid/ebird`) on first run;
2. refreshes the curated seed files when the installed package version changes
   (tracked by the `.seed-version` stamp written by the PKGBUILD), while never
   touching user-downloaded data (`regions_generated/`, `eBird_Taxonomy_v2023.csv`);
3. exports `BIRDID_DATA_DIR` (the override added in `birdid.py`) and execs the
   real launcher.

## Local test / build

The real AUR sources live in `~/aur/{birdid-git,python-pybioclip}`. To build:

```bash
sudo pacman -S --needed python-build python-installer python-wheel namcap
cd ~/aur/python-pybioclip && makepkg -f          # builds + .SRCINFO
cd ~/aur/birdid-git   && makepkg -f              # needs the GitHub repo public
```

`birdid-git` needs the GitHub repo to be public (the `git+https` source URL).
Until then it is validated with a local-source PKGBUILD variant
(`git+file://...`) plus a fake-root run of the wrapper.
