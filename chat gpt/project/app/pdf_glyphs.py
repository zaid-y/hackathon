"""Repair font-specific Thai PUA mappings using embedded glyph outlines.

Only exact outlines (ignoring placement) with an unambiguous standard Thai
counterpart in the same PDF/font family are accepted. Unknown glyphs survive.
The source PDF is never written: only the reader's in-memory ToUnicode changes.
"""
from io import BytesIO
import hashlib
import re

from fontTools.ttLib import TTFont
from pypdf.generic import DecodedStreamObject, NameObject

_PAIR = re.compile(rb"<([0-9a-fA-F]{4})>\s*<([0-9a-fA-F]{4})>")
_BLOCK = re.compile(rb"\d+\s+beginbfchar\b(.*?)endbfchar", re.S)

# Exact outline hashes verified against standard-Unicode glyphs embedded in
# AIT, DSBA, IT2565 and IT_inter2565. Some individual PDFs omit the standard
# counterpart. Never use a global PUA-to-letter table: codes differ by subset.
_VERIFIED = {
    # Visually verified compact bold mai tho variant (same two contours).
    '52591eb21a9c374fa82aa681f7eb9b019de4c893d49cbc745c3740a890eb230c': 3657,
    '0fa641fde2abff437f3b6b436c4b98f853be57ab1c7cd62183a27b98e39458b7': 3638,
    '12945c6f46081a276f80c6ae37c292edaa2c14e60820555a62a4f6e3f453cf73': 3655,
    '2b0e509438d7e90414897d5a69271862ccfdbba78048b51573c45d1cce83c99a': 3656,
    '46fc02597ea4b50954d13883e488d418a0a7510452545cdd6cac9df2362d9e56': 3656,
    '4b34da2738d35073e75e16b0c04ef7977b2a9fc697ecef87be56a37f44d212a5': 3660,
    '50ecc347d4bd017eafede210fbe86c32e2442af5efd0e74710b9e65014086c61': 3657,
    '59e068551c8942ba01ad958d3ebce83910c5986ca346113fdb169f4471ed05b9': 3656,
    '6c96d002b83b0cb3db40753608bb657243cfcebdc47b9f49caf47c217a5bf6ef': 3633,
    '6df34488185ff36431c0e89568186aef16e59945f3776964af5dd412bba7aac6': 3633,
    '7a9d3298c027f8542dd4bd56937cfb6019b7104771846ee2d917bb68c96f1b6e': 3657,
    '86cee227bdd8fbe51e29dbaf69a35355646faaf2ba673de356ea832db54c5782': 3656,
    '8c756db1fb4ebeb2afb8006a7a29a3963f88df521673f8cf73799c5984f171f3': 3655,
    '907b462336fbbd28cc61a4891ecbd41727afb17ef062abadcffcc95b9bff12d0': 3660,
    '9269bc18f766a51fa32ccefc414e226a687d2f9163bf58d783041252cc3069c4': 3657,
    '9b592591bd8f6dc820e8ef30dd749c433f4f3a0fdd9cb43973dd6afa2b433b5e': 3636,
    '9df82d6fe72a11d96127b62ce5888770072e749d663d8ea99ecd1b94686bb26f': 3658,
    'a824d6a8fefe1f181ed790bb2d004a956ffd08b8a2ae7b43ee1c07c33262f0ca': 3636,
    'b59c0cac6684a69aa4d89295e8f82a8bb721c5452c7f105b6a4d5ceb2110c1ba': 3637,
    'bd6e72332da0131c54cfa418a0eebdc172e482a6417fdb920a1da5be601e33d1': 3660,
    'd8cfb30ffa7be69d42479be7fb3f9b13104f03e7238b110c086434a79f0afd04': 3660,
    'e04418b746fbb6bcfad247b212a2cecb2e25545dd530fa05914f3b843b3d74ce': 3637,
    'ec9470ea3bfd0f6a4d02f0117d4204c3f357e9904466a7f493f2acf90daf31b3': 3657,
    'f37d77d776086689a3de1ff0e8d27b71f74bc0901ccf001fe58f6a45b57dd77a': 3638,
}


def _fonts(resources, seen):
    for ref in resources.get('/Font', {}).get_object().values() if '/Font' in resources else ():
        font = ref.get_object()
        if id(font) not in seen:
            seen.add(id(font))
            yield font
    for ref in resources.get('/XObject', {}).get_object().values() if '/XObject' in resources else ():
        obj = ref.get_object()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if '/Resources' in obj:
            yield from _fonts(obj['/Resources'], seen)


def _outline(font, gid):
    if not 0 <= gid < len(font.getGlyphOrder()):
        return None
    coords, ends, flags = font['glyf'][font.getGlyphName(gid)].getCoordinates(font['glyf'])
    if not coords:
        return None
    x, y = coords[0]
    return tuple((a-x, b-y) for a, b in coords), tuple(ends), tuple(flags)


def _unique_target(candidates):
    return next(iter(candidates)) if len(candidates) == 1 else None


def repair_thai_glyphs(reader):
    references, pending, seen = {}, [], set()
    for page in reader.pages:
        for pdf_font in _fonts(page.get('/Resources', {}), seen):
            family = str(pdf_font.get('/BaseFont', '')).split('+')[-1]
            if not family.startswith('THSarabun') or '/DescendantFonts' not in pdf_font or '/ToUnicode' not in pdf_font:
                continue
            descendant = pdf_font['/DescendantFonts'][0].get_object()
            descriptor = descendant.get('/FontDescriptor')
            if not descriptor or '/FontFile2' not in descriptor.get_object():
                continue
            # Unsupported/malformed embedded fonts must not prevent extraction.
            try:
                font = TTFont(BytesIO(descriptor.get_object()['/FontFile2'].get_data()))
                data = pdf_font['/ToUnicode'].get_data()
                gid_map = descendant.get('/CIDToGIDMap', '/Identity')
                gid_bytes = None if gid_map == '/Identity' else gid_map.get_object().get_data()
                entries = []
                for block in _BLOCK.finditer(data):
                    for match in _PAIR.finditer(block.group(1)):
                        cid, codepoint = (int(v, 16) for v in match.groups())
                        gid = cid if gid_bytes is None else int.from_bytes(gid_bytes[2*cid:2*cid+2], 'big')
                        outline = _outline(font, gid)
                        if outline is None:
                            continue
                        key = family, outline
                        if 0x0e01 <= codepoint <= 0x0e4e:
                            references.setdefault(key, set()).add(codepoint)
                        elif 0xe000 <= codepoint <= 0xf8ff:
                            known = _VERIFIED.get(hashlib.sha256(repr(outline).encode()).hexdigest())
                            if known is not None:
                                references.setdefault(key, set()).add(known)
                            entries.append((cid, key))
                pending.append((pdf_font, data, entries))
                font.close()
            except (KeyError, ValueError, IndexError, AssertionError):
                continue
    repaired = 0
    for pdf_font, data, entries in pending:
        replacements = {cid: target for cid, key in entries
                        if (target := _unique_target(references.get(key, set()))) is not None}
        if not replacements:
            continue
        def block_replace(block):
            def pair_replace(match):
                nonlocal repaired
                target = replacements.get(int(match[1], 16))
                if target is None:
                    return match[0]
                repaired += 1
                return b'<' + match[1] + b'> <' + f'{target:04X}'.encode() + b'>'
            return block[0].replace(block[1], _PAIR.sub(pair_replace, block[1]), 1)
        stream = DecodedStreamObject()
        stream.set_data(_BLOCK.sub(block_replace, data))
        pdf_font[NameObject('/ToUnicode')] = stream
    return repaired
