#!/usr/bin/env python3
"""
/**
 * This file is licensed under the European Union Public License (EUPL) v1.2.
 * You may only use this work in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed "as is",
 * without any warranty or conditions of any kind.
 *
 * Copyright (c) 2024- Andre Karlsson. All rights reserved.
 *
 * Created on 5/30/25 :: 09:59 BY joyider <andre(-at-)sess.se>
 *
 * This file :: get_base_names.py is part of the cool_name_generator project.
 */
 """
import requests
import json
import time
import re
import os
import ast

API_URL = "https://en.wikipedia.org/w/api.php"
CATEGORIES = [
    "The_Lord_of_the_Rings_characters",
    "A_Song_of_Ice_and_Fire_characters",
    "Harry Potter characters",
    "The Wheel of Time characters",
    "Discworld characters",
    "The Chronicles of Narnia characters",
    "Dragonlance characters",
    "The_Witcher_character_redirects_to_lists",
    "Earthsea characters",
    "Stormlight Archive characters",
    "The Hobbit characters",
    "His Dark Materials characters",
    "Malazan Book of the Fallen characters",
    "The Chronicles of Amber characters",
    "Kingkiller Chronicle characters",
    "Shannara characters",
    "The Dark Tower character redirects to lists"
]

def get_category_members(category, max_results=500):
    members = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": max_results
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = requests.get(API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        for m in data["query"]["categorymembers"]:
            members.append(m["title"])
        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
            time.sleep(0.2)
        else:
            break
    return members

import re

import re

def clean_and_split(name):
    if (
        re.search(r'\d', name)
        or (name and name[0].islower())
        or any(tok in name for tok in ("printscreen", "Fellowship", "movie", "Movie", "List", "Category", "(", ")"))
    ):
        return [] 

    name = re.sub(r'^File:', '', name, flags=re.IGNORECASE) 

    name = re.sub(r'\.(?:jpg|jpeg|gif|png)$', '', name, flags=re.IGNORECASE)

    parts = re.split(r'[ \-]+', name)
    cleaned = []
    
    for p in parts:
        p = p.strip().rstrip(".,;:")
        if not p:
            continue
        if p[0].islower():
            continue

    subparts = re.findall(r'[A-ZÅÄÖ][^A-ZÅÄÖ]*', p)
    if not subparts:
        subparts = [p]

    for sp in subparts:
        sp = sp.strip().rstrip(".,;:")
        if len(sp) > 3:
            cleaned.append(sp)
    
    return cleaned

def main():
    all_names = set()

    for cat in CATEGORIES:
        print(f"Fetching category: {cat}…")
        try:
            raw = get_category_members(cat)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        for entry in raw:
            tokens = clean_and_split(entry)
            for t in tokens:
                all_names.add(t)
        print(f"  Total unique fetched so far: {len(all_names)}")

    final = list(all_names)

    art_path = os.path.join('.', 'artistic_names')
    det_path = os.path.join('.', 'deterministic_names')
    holistic_path = os.path.join('.', 'holistic_names')

    with open(art_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    with open(det_path, 'r', encoding='utf-8') as f:
        det_content = f.read().strip()
    with open(holistic_path, 'r', encoding='utf-8') as f:
        hol_content = f.read().strip()

    _, list_literal = content.split('=', 1)
    _, det_list_literal = det_content.split('=', 1)
    _, hol_list_literal = hol_content.split('=', 1)

    artistic_names = ast.literal_eval(list_literal)
    det_names = ast.literal_eval(det_list_literal)
    hol_names = ast.literal_eval(hol_list_literal)

    unique = list(set(
        re.sub(r'(.)\1{2,}', r'\1\1', name)
        for name in final + artistic_names + det_names + hol_names
    ))

    out = {
        "base_names": unique
    }

    out_dir = os.path.join('..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'base_names.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(unique)} to base_names.json")

if __name__ == "__main__":
    main()


