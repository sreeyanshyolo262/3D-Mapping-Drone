#!/usr/bin/env python3
"""Make an RTAB-Map database exportable when its LAST frame is a stray, unconnected one.

rtabmap-export builds the cloud from whatever is connected to the last frame. RTAB-Map sometimes
writes a lone frame at shutdown (weight -9, no links); then the export contains just that one
camera image (~180,000 points) and the rest of the scan is silently left out.

This copies the database and deletes such trailing frames from the COPY (the original is never
changed), so the export starts from the real map.

Usage: python3 fix_last_node.py <in.db> <out.db>
Exit code: 0 = a fixed copy was written, 3 = nothing needed fixing (use the original), 1 = error.
Only needs Python's standard library.
"""
import shutil
import sqlite3
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(src)
    removed = []
    try:
        for _ in range(8):
            last = con.execute("select max(id) from Node").fetchone()[0]
            if last is None:
                break
            links = con.execute("select count(*) from Link where from_id=? or to_id=?",
                                (last, last)).fetchone()[0]
            if links:
                break
            removed.append(last)
            # pretend it is gone for the next loop pass without touching the original file
            con.execute("create temp table if not exists gone(id integer)")
            con.execute("insert into gone values (?)", (last,))
            nxt = con.execute("select max(id) from Node where id not in (select id from gone)").fetchone()[0]
            if nxt is None:
                break
            links = con.execute("select count(*) from Link where from_id=? or to_id=?",
                                (nxt, nxt)).fetchone()[0]
            if links:
                break
        con.close()
    except sqlite3.Error as e:
        print("fix_last_node: cannot read the database:", e)
        return 1
    if not removed:
        print("fix_last_node: last frame is connected, nothing to fix")
        return 3

    shutil.copyfile(src, dst)
    db = sqlite3.connect(dst)
    try:
        for n in removed:
            db.execute("delete from Node where id=?", (n,))
            db.execute("delete from Data where id=?", (n,))
            db.execute("delete from Feature where node_id=?", (n,))
            db.execute("delete from Statistics where id=?", (n,))
            db.execute("delete from GlobalDescriptor where node_id=?", (n,))
        # the saved "optimised poses" may list the removed frames; let RTAB-Map recompute them
        db.execute("update Admin set opt_ids=NULL, opt_poses=NULL")
        db.commit()
        left = db.execute("select max(id) from Node").fetchone()[0]
    except sqlite3.Error as e:
        print("fix_last_node: could not edit the copy:", e)
        db.close()
        return 1
    db.close()
    print("fix_last_node: removed stray last frame(s) %s; the map now ends at frame %s -> %s"
          % (", ".join(map(str, sorted(removed))), left, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
