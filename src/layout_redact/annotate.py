"""Minimal template authoring GUI (tkinter, ships with Python).

    layout-redact annotate reference.png -o my-form.yaml [-t existing.yaml]

* drag with the left mouse button to draw a redaction box,
* type a name for it (or accept the default),
* select a region in the list and press Delete to remove it,
* the green dots are the line junctions that will become template keypoints,
* Save writes the template (keypoints + frame + regions).

Coordinates are stored in reference-image pixels, whatever the zoom.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .io import load_image
from .skeleton import extract_skeleton
from .template import Region, Template, build_template


def run_annotator(reference: str, output: str, template_path: Optional[str] = None,
                  name: Optional[str] = None, dpi: int = 150, max_display: int = 900) -> int:
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    ref = load_image(reference, dpi=dpi)
    h, w = ref.shape[:2]
    regions: List[Region] = []
    if template_path:
        old = Template.load(template_path)
        if tuple(old.page_size) != (w, h):
            raise SystemExit(f"Template page size {old.page_size} != reference size {(w, h)}")
        regions = list(old.regions)
        name = name or old.name
    name = name or Path(reference).stem

    sk = extract_skeleton(ref)
    scale = min(1.0, max_display / max(h, w))
    disp = cv2.resize(ref, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    for j in sk.junctions:
        cv2.circle(disp, (int(j.x * scale), int(j.y * scale)), 3, (40, 170, 40), -1)

    root = tk.Tk()
    root.title(f"layout-redact annotate - {name}")
    photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
    canvas = tk.Canvas(root, width=photo.width(), height=photo.height(), cursor="crosshair")
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.grid(row=0, column=0, rowspan=3)
    listbox = tk.Listbox(root, width=32)
    listbox.grid(row=0, column=1, sticky="ns", padx=6, pady=6)
    status = tk.Label(root, text=f"{len(sk.junctions)} junctions detected. Drag to add a region.",
                      anchor="w", justify="left", wraplength=230)
    status.grid(row=1, column=1, sticky="we", padx=6)

    drag = {"start": None, "rect": None}
    shapes: List[int] = []

    def redraw():
        for s in shapes:
            canvas.delete(s)
        shapes.clear()
        listbox.delete(0, tk.END)
        for r in regions:
            pts = (r.polygon * scale).ravel().tolist()
            shapes.append(canvas.create_polygon(*pts, outline="#ff7a00", fill="", width=2))
            x, y = (r.polygon.min(axis=0) * scale).tolist()
            shapes.append(canvas.create_text(x + 3, y + 3, text=r.name, anchor="nw", fill="#d45500"))
            listbox.insert(tk.END, r.name)

    def on_press(e):
        drag["start"] = (e.x, e.y)
        drag["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#0077ff", width=2)

    def on_move(e):
        if drag["rect"] is not None:
            x0, y0 = drag["start"]
            canvas.coords(drag["rect"], x0, y0, e.x, e.y)

    def on_release(e):
        if drag["rect"] is None:
            return
        x0, y0 = drag["start"]
        canvas.delete(drag["rect"])
        drag["rect"] = None
        if abs(e.x - x0) < 4 or abs(e.y - y0) < 4:
            return
        default = f"region_{len(regions) + 1}"
        nm = simpledialog.askstring("Region name", "Name for this region:", initialvalue=default,
                                    parent=root)
        if not nm:
            return
        box = np.array([x0, y0, e.x, e.y], np.float64) / scale
        regions.append(Region.from_box(nm.strip(), box))
        redraw()

    def on_delete(_e=None):
        sel = listbox.curselection()
        if sel:
            regions.pop(sel[0])
            redraw()

    def on_save():
        try:
            tpl = build_template(ref, regions, name=name)
        except ValueError as exc:
            messagebox.showerror("Cannot build template", str(exc))
            return
        tpl.save(output)
        status.config(text=f"Saved {len(regions)} regions, {len(tpl.keypoints)} keypoints to {output}")

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    listbox.bind("<Delete>", on_delete)
    buttons = tk.Frame(root)
    buttons.grid(row=2, column=1, sticky="we", padx=6, pady=6)
    tk.Button(buttons, text="Delete selected", command=on_delete).pack(fill="x")
    tk.Button(buttons, text="Save template", command=on_save).pack(fill="x", pady=(4, 0))
    redraw()
    root.mainloop()
    return 0
