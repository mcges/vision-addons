#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backup de los pesos de Chatterbox en GitHub (canal publico `mcges/vision-addons`).

POR QUE EXISTE
--------------
El add-on de voz NO lleva los pesos (~3,2 GB): los baja la primera vez. Fuente normal =
HuggingFace. Este script crea una copia ESPEJO en un Release publico de GitHub para que el
add-on siga funcionando si algun dia HuggingFace deja de servirlos. El add-on comprueba el
SHA-256 de cada fichero, asi que el backup es verificable de punta a punta.

LIMITE DE GITHUB: 2 GB por asset -> los ficheros grandes se suben TROCEADOS
(`<nombre>.part01`, `.part02`, ...) y el add-on los concatena y verifica el hash final.
El manifiesto (`manifest.json`) es lo que lee `companion/addons-src/clon/run.py::backup_descargar`.

USO
---
  # 1) construir y subir al Release publico (lo que hace el workflow)
  python voice_model_backup.py --out dist --variante ambos --subir mcges/vision-addons voice-model-chatterbox-1

  # 2) solo construir en local (sin subir)
  python voice_model_backup.py --out dist --variante multilingue

  # 3) probar el troceado/reensamblado sin red ni ficheros de 2 GB
  python voice_model_backup.py --autotest

NO toca el plugin ni el companion: solo genera ficheros + manifiesto y (si se pide) sube
assets con `gh`. Nunca sobrescribe nada fuera de `--out`.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time

# Ficheros EXACTOS por variante (mismos que `MODEL_FILES` en run.py; upstream: tts.py y mtl_tts.py).
MODEL_REPO = 'ResembleAI/chatterbox'
MODEL_FILES = {
    'base': ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt'],
    'multilingue': ['ve.pt', 't3_mtl23ls_v2.safetensors', 's3gen.pt',
                    'grapheme_mtl_merged_expanded_v1.json', 'conds.pt', 'Cangjie5_TC.json'],
}
# 1,9 GB: por debajo del tope de 2 GB por asset de GitHub (y con margen).
PARTE_MAX_BYTES = 1900000000
BLOQUE = 1024 * 1024


def log(msg):
    sys.stderr.write('[backup] ' + str(msg) + '\n')
    sys.stderr.flush()


def sha256_file(ruta):
    h = hashlib.sha256()
    with open(ruta, 'rb') as fh:
        while True:
            trozo = fh.read(BLOQUE)
            if not trozo:
                break
            h.update(trozo)
    return h.hexdigest()


def trocear(ruta, destino_dir, max_bytes=PARTE_MAX_BYTES, nombre=None):
    """Parte `ruta` en trozos de `max_bytes`. Devuelve la lista de partes del manifiesto.

    El fichero original NO se borra aqui (lo decide quien llama): el hash del fichero completo
    se calcula antes, durante el troceado. `nombre` permite forzar el nombre base de las partes
    (para no ensuciar el asset con el sufijo del fichero temporal).
    """
    nombre = nombre or os.path.basename(ruta)
    partes = []
    with open(ruta, 'rb') as fh:
        indice = 0
        while True:
            trozo = fh.read(max_bytes)
            if not trozo:
                break
            indice += 1
            pn = nombre + ('.part%02d' % indice)
            pruta = os.path.join(destino_dir, pn)
            with open(pruta, 'wb') as salida:
                salida.write(trozo)
            partes.append({'name': pn, 'size': len(trozo), 'sha256': sha256_file(pruta)})
    return partes


def descargar_hf(nombre, revision=''):
    """Baja UN fichero del repo oficial de HuggingFace (sin traer el repo entero)."""
    from huggingface_hub import hf_hub_download  # noqa: WPS433 - dependencia opcional
    args = {'repo_id': MODEL_REPO, 'filename': nombre}
    if revision:
        args['revision'] = revision
    return hf_hub_download(**args)


def construir(variante, out_dir, revision=''):
    """Baja y prepara los assets de una variante. Devuelve (entrada_manifiesto, ok)."""
    entrada = {'files': []}
    os.makedirs(out_dir, exist_ok=True)
    for nombre in MODEL_FILES[variante]:
        ruta = descargar_hf(nombre, revision)
        size = os.path.getsize(ruta)
        log('%s: %.1f MB (%s)' % (nombre, size / 1048576.0, variante))
        digest = sha256_file(ruta)
        registro = {'name': nombre, 'size': size, 'sha256': digest, 'parts': []}
        if size > PARTE_MAX_BYTES:
            # Troceado: se copia a `out_dir` por partes y el original se descarta (disco del CI).
            # Las partes se llaman `<nombre>.partNN` (limpio) y se borra el temporal al terminar.
            destino = os.path.join(out_dir, nombre + '.fuente')
            shutil.copy2(ruta, destino)
            registro['parts'] = trocear(destino, out_dir, PARTE_MAX_BYTES, nombre=nombre)
            os.remove(destino)
            log('  troceado en %d partes (%s)' % (len(registro['parts']), nombre))
        else:
            shutil.copy2(ruta, os.path.join(out_dir, nombre))
        entrada['files'].append(registro)
    return entrada, True


def limpiar_cache_hf():
    """Borra la cache de HuggingFace del runner (los ficheros ya estan copiados en `out`).

    Necesario para respaldar AMBAS variantes en un runner de ~14 GB: 3,2 GB de cache + 3,2 GB
    de assets por variante no caben dos veces.
    """
    base = os.path.expanduser('~/.cache/huggingface')
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
        log('cache de HuggingFace liberada (' + base + ')')


def subir(out_dir, repo, tag, notas_extra=''):
    """Crea el Release (si falta) y sube los assets con `gh` (idempotente con --clobber)."""
    def gh(*args, **kw):
        cmd = ['gh'] + list(args)
        log('gh ' + ' '.join(cmd[1:]))
        return subprocess.run(cmd, check=kw.get('check', True))

    try:
        gh('release', 'view', tag, '-R', repo)
    except subprocess.CalledProcessError:
        notas = ('Backup del modelo de voz (Chatterbox) para el add-on de clonacion de Vision IA.\n\n'
                 '- Repo original: https://huggingface.co/%s\n'
                 '- Los ficheros de mas de 2 GB van TROCEADOS (`.partNN`): el add-on los concatena\n'
                 '  y verifica el SHA-256 final del manifiesto antes de usar nada.\n'
                 '- El add-on solo usa este backup si HuggingFace no responde.\n%s' % (MODEL_REPO, notas_extra))
        gh('release', 'create', tag, '-R', repo, '--title', 'Modelo de voz (backup)', '--notes', notas)
    nombres = sorted(os.listdir(out_dir))
    if not nombres:
        raise SystemExit('no hay assets que subir en ' + out_dir)
    # En tandas: la linea de comandos no puede crecer sin limite.
    tanda = []
    for n in nombres:
        tanda.append(os.path.join(out_dir, n))
        if len(tanda) >= 8:
            gh('release', 'upload', tag, *tanda, '-R', repo, '--clobber')
            tanda = []
    if tanda:
        gh('release', 'upload', tag, *tanda, '-R', repo, '--clobber')


def autotest():
    """Prueba el troceado + reensamblado con un fichero sintetico (sin red)."""
    import tempfile
    base = tempfile.mkdtemp(prefix='backup_autotest_')
    try:
        origen = os.path.join(base, 'fake.safetensors')
        datos = os.urandom(3000)
        with open(origen, 'wb') as fh:
            fh.write(datos)
        partes = trocear(origen, base, max_bytes=1000)
        assert len(partes) == 3, 'esperaba 3 partes, hay %d' % len(partes)
        reensamblado = os.path.join(base, 'reensamblado')
        with open(reensamblado, 'wb') as salida:
            for parte in partes:
                with open(os.path.join(base, parte['name']), 'rb') as fh:
                    salida.write(fh.read())
        assert open(reensamblado, 'rb').read() == datos, 'el reensamblado no coincide'
        assert sha256_file(reensamblado) == sha256_file(origen), 'los hashes no coinciden'
        for parte in partes:
            assert sha256_file(os.path.join(base, parte['name'])) == parte['sha256'], 'hash de parte mal'
        log('autotest OK (3 partes, reensamblado y hashes correctos)')
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description='Backup de los pesos de Chatterbox en GitHub.')
    ap.add_argument('--out', default='dist', help='carpeta de salida de los assets')
    ap.add_argument('--variante', default='ambos', choices=['ambos', 'base', 'multilingue'])
    ap.add_argument('--revision', default='', help='revision de HuggingFace (por defecto, la actual)')
    ap.add_argument('--subir', nargs=2, metavar=('REPO', 'TAG'),
                    help='sube los assets al Release (p. ej. mcges/vision-addons voice-model-chatterbox-1)')
    ap.add_argument('--autotest', action='store_true', help='prueba troceado/reensamblado sin red')
    ap.add_argument('--limpiar-cache', action='store_true',
                    help='borra la cache de HuggingFace tras subir cada variante (disco del CI)')
    args = ap.parse_args()

    if args.autotest:
        return autotest()

    variantes = ['base', 'multilingue'] if args.variante == 'ambos' else [args.variante]
    manifiesto = {
        'repo': MODEL_REPO,
        'revision': args.revision or '',
        'generado': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'parte_max_bytes': PARTE_MAX_BYTES,
        'variantes': {},
    }
    out = os.path.abspath(args.out)
    for variante in variantes:
        log('== variante ' + variante + ' ==')
        entrada, _ = construir(variante, out, args.revision)
        manifiesto['variantes'][variante] = entrada
        with io.open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(manifiesto, indent=2, ensure_ascii=False))
        if args.subir:
            subir(out, args.subir[0], args.subir[1], '\n- Variantes incluidas: ' + ', '.join(variantes))
            # Se libera disco tras subir cada variante (el CI tiene ~14 GB).
            for n in sorted(os.listdir(out)):
                if n == 'manifest.json':
                    continue
                try:
                    os.remove(os.path.join(out, n))
                except OSError:
                    pass
            if args.limpiar_cache:
                limpiar_cache_hf()
    if args.limpiar_cache:
        limpiar_cache_hf()
    log('manifiesto: ' + os.path.join(out, 'manifest.json'))
    log('variantes: ' + ', '.join(manifiesto['variantes'].keys()))
    log('ficheros por variante: ' + json.dumps({k: len(v['files']) for k, v in manifiesto['variantes'].items()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
