"""Labelled-image dataset contract for adapting the feature extractor: the pinned iNaturalist bird sample,
validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and out of the checkpoint's self-supervised distribution: 180 CC0-licensed,
research-grade iNaturalist photographs of six common North American birds (30 per species, one per observer
per species), chosen a priori on 2026-09-19 and pinned here by photo id, byte size and SHA-256 of the served
`medium` JPEG. Every file is fetched from the iNaturalist open-data bucket at run time and refused on any
byte-size or SHA-256 mismatch; the repository redistributes none of the photographs. Each record keeps the
observation id and observer login so every image is traceable to its public observation page.

A record is ``{id, image, label}``: a PIL image (or a path to one) and the species key of its gold label.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MAX_IMAGE_SIDE, MODEL_ID

CORPUS_NAME = "iNaturalist CC0 bird photographs (six species)"
CORPUS_RELEASE = "iNaturalist open-data bucket, research-grade CC0 photos selected 2026-09-19"
CORPUS_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
CORPUS_LICENSE = "CC0 1.0 (each photo's own license_code on iNaturalist; observers credited in the records)"
CORPUS_BYTES = 19_183_071
DEFAULT_CACHE_DIR = Path("weights") / "inat-birds"
SPECIES: dict[str, tuple[str, str]] = {
    "song_sparrow": ("Melospiza melodia", "Song Sparrow"),
    "chipping_sparrow": ("Spizella passerina", "Chipping Sparrow"),
    "white_throated_sparrow": ("Zonotrichia albicollis", "White-throated Sparrow"),
    "dark_eyed_junco": ("Junco hyemalis", "Dark-eyed Junco"),
    "house_finch": ("Haemorhous mexicanus", "House Finch"),
    "american_goldfinch": ("Spinus tristis", "American Goldfinch"),
}
# (id, label, iNat photo id, iNat observation id, observer login, bytes, sha256 of <photo id>/medium.jpg)
SAMPLE_RECORDS: tuple[tuple[str, str, int, int, str, int, str], ...] = (
    (
        "song_sparrow-00",
        "song_sparrow",
        129376982,
        79016324,
        "andywilson",
        43427,
        "7a9d9304a82f202e992655ec5f65477cd3d7c1dce03aa89a214c2daa38f9d61d",
    ),
    (
        "song_sparrow-01",
        "song_sparrow",
        480991086,
        267636534,
        "lyneisfilm",
        162073,
        "11f77ff277dd2703c1000f2c787136ff0c3ca7ffad7017056892fe789d65efec",
    ),
    (
        "song_sparrow-02",
        "song_sparrow",
        546060381,
        302980489,
        "swpollinators",
        27899,
        "4e70b9519c6e5f7384a4495b91b45465f2b1599f86491d8f9f9b12635a4046f6",
    ),
    (
        "song_sparrow-03",
        "song_sparrow",
        308625896,
        177450028,
        "radrat",
        70961,
        "1211da4fdb24ae85ef0c6c3e2d03542c430457856aec661fe8f5f2de0027eee5",
    ),
    (
        "song_sparrow-04",
        "song_sparrow",
        494793016,
        275349085,
        "k-simpkins",
        58410,
        "255538cf450197257e86ed3d41dc69fb78e594434e9cb338c6314288c6cff26e",
    ),
    (
        "song_sparrow-05",
        "song_sparrow",
        674054489,
        369029444,
        "ben142",
        220573,
        "1ae24622888d9d449ffd6b5c65cac1b9b14870fd8e12dbed8aa0acc2f5030123",
    ),
    (
        "song_sparrow-06",
        "song_sparrow",
        339623726,
        193339933,
        "rawcomposition",
        25012,
        "d2cde085277a71886a2bf211a1eec26941a73752375708726a8af29aa4995a8b",
    ),
    (
        "song_sparrow-07",
        "song_sparrow",
        222768957,
        130949329,
        "davidfbird",
        110773,
        "0feee62753f409d0aa365e9aa017ddef436ad70a2673dc847feb3b5386af27ff",
    ),
    (
        "song_sparrow-08",
        "song_sparrow",
        181658744,
        107953669,
        "gcart043",
        98482,
        "a662a6abb24f42b256fb6e2d6f02c3e128a1053ef534f454461aa5cadcc03fe4",
    ),
    (
        "song_sparrow-09",
        "song_sparrow",
        148994242,
        90171417,
        "glennberry",
        102702,
        "64333977d24957d723a1d26886004f222d810557617685579f72a6b553e508fa",
    ),
    (
        "song_sparrow-10",
        "song_sparrow",
        637315932,
        349374463,
        "sooji",
        136572,
        "a5cebbc0cc2325d3805c4ac854e103f7ba1c22e3a34fb204f30d58681e865033",
    ),
    (
        "song_sparrow-11",
        "song_sparrow",
        471146686,
        262252507,
        "jeanpaulboerekamps",
        98601,
        "4301f06b52b8dcf1e137567c32412e384edb71cb90e2e35459a0405b2e56529b",
    ),
    (
        "song_sparrow-12",
        "song_sparrow",
        123859010,
        75689904,
        "w_mark_c",
        190819,
        "6b6057a1c50b83ffeb4d9e34367b3e6a9b355236dbcae9820b509e1484f8fe33",
    ),
    (
        "song_sparrow-13",
        "song_sparrow",
        640811426,
        351179648,
        "erikschiff",
        107695,
        "27aecce184a485ee888c0e8101cb4a34899b8a185b62dc064f3dc2ec9902182b",
    ),
    (
        "song_sparrow-14",
        "song_sparrow",
        63537800,
        40010230,
        "nathanael15",
        51441,
        "5ae55f868e779a4e8ee34f6ad077b40c41f20aa699d967f373ebd659a89137a8",
    ),
    (
        "song_sparrow-15",
        "song_sparrow",
        108520869,
        67204020,
        "dugald",
        52496,
        "e483364889fb95c540db84b5edf5a2400febf7d2eb62a1e49303b13ad329d1b8",
    ),
    (
        "song_sparrow-16",
        "song_sparrow",
        435251292,
        244042351,
        "carterdorscht",
        166662,
        "d0cdd9ddcf7202a91ac2c47910a0c230639bce50283e8511fb8311151763233a",
    ),
    (
        "song_sparrow-17",
        "song_sparrow",
        120710033,
        73898230,
        "tys_rbg",
        126820,
        "515b3b32b64e88d401990bfd1d8e2c1281443b5d1e763a09e86d3d3664e1df41",
    ),
    (
        "song_sparrow-18",
        "song_sparrow",
        393408927,
        222067370,
        "irenemacaulay_",
        101681,
        "22e326807de963352b4307790bd40c5506adb0fab1f9844968edaf4bc5665e1c",
    ),
    (
        "song_sparrow-19",
        "song_sparrow",
        349361283,
        198243065,
        "sean579",
        42820,
        "5eb0031a8d6066650c66b265fb1724413273e095e4e531054d2817eb75910074",
    ),
    (
        "song_sparrow-20",
        "song_sparrow",
        608802621,
        335154467,
        "jamesadney",
        112564,
        "6e94bce1b5135f48b0b19b64e21a0c76cff4f9b9a28fcbc2b7f5c8ddda6ef513",
    ),
    (
        "song_sparrow-21",
        "song_sparrow",
        614258628,
        337847585,
        "joy4birds",
        70552,
        "aa767aa5a74a76cfd985aba5282e589f670a9ce0a8fcdc6a096c0357992dec67",
    ),
    (
        "song_sparrow-22",
        "song_sparrow",
        634100352,
        347744524,
        "zorthesosen",
        214044,
        "6a70dba26dfb3e98a244a9ec9badb812ddaa8b3612e7a36f66aba057bb757ecd",
    ),
    (
        "song_sparrow-23",
        "song_sparrow",
        50065474,
        31954532,
        "truthseqr",
        82521,
        "4186fbf344e92038358d4338102aa440098bff5f558ab1d197f72fefbcec0b15",
    ),
    (
        "song_sparrow-24",
        "song_sparrow",
        8656044,
        6803564,
        "glmory",
        297964,
        "e8cab773436ccfaad4699e5112ef4edb516236d413d6dbf034eea7bf9c88c8f5",
    ),
    (
        "song_sparrow-25",
        "song_sparrow",
        131051149,
        79988590,
        "funvill",
        72572,
        "0a694b5bb03aee6eb5a3a6132e855d47172b21b84cc4cc7c64367d1a31de7894",
    ),
    (
        "song_sparrow-26",
        "song_sparrow",
        12923156,
        9491600,
        "gambolingquail",
        69559,
        "6a92bff4fc76820c6f21e15c5f254385dd976a32264911bfc08417a1c2bf053d",
    ),
    (
        "song_sparrow-27",
        "song_sparrow",
        12077500,
        8959545,
        "reuvenm",
        74010,
        "3e3c5e94c839f45610ef3ef7ffaf3575f4bfe365fc94454c30bfdb718ca3c593",
    ),
    (
        "song_sparrow-28",
        "song_sparrow",
        132730299,
        80955309,
        "steph123456",
        154309,
        "5f8854dd231a302c643b22521b49ed3007203d74b8b8413a23abc79ffe0b0270",
    ),
    (
        "song_sparrow-29",
        "song_sparrow",
        124840110,
        76316566,
        "terrimewbornagain",
        81046,
        "2020092b0b67397a78cc2b0df267fb671b3ba18e03fb0e35e664e0784061e3c6",
    ),
    (
        "chipping_sparrow-00",
        "chipping_sparrow",
        198992636,
        117809422,
        "k-simpkins",
        62563,
        "cf9f3b0c1863808e21af596b2e609b047ddbc28cb2ed076625e2546425ff0adf",
    ),
    (
        "chipping_sparrow-01",
        "chipping_sparrow",
        248210057,
        144599194,
        "w_mark_c",
        193389,
        "497d0a0fef81c326bcc87b5d1eb97fe987d559b8f2b71bb60fe422ae34dce150",
    ),
    (
        "chipping_sparrow-02",
        "chipping_sparrow",
        156350853,
        94266719,
        "ellyne",
        142332,
        "6a60ebac34476a372b4790a87d823432cdda8f72590930b5d18ae166cf4c7ba2",
    ),
    (
        "chipping_sparrow-03",
        "chipping_sparrow",
        16128796,
        11327134,
        "reuvenm",
        70532,
        "5ad36c9cdd6c92e225a1b8ab3c04d2f65bc4e971d0243958f8ac090b0996115c",
    ),
    (
        "chipping_sparrow-04",
        "chipping_sparrow",
        391300648,
        220982684,
        "carterdorscht",
        208567,
        "c974a676c3c227c2844ff822d429c784bc87bb41f40d5074a0ebadd4c6785b21",
    ),
    (
        "chipping_sparrow-05",
        "chipping_sparrow",
        339456083,
        193252042,
        "rawcomposition",
        46329,
        "41c9260ca9107430e3a8090ba01cebf3e3c25f2dad15bea4f77c8e824d9fc496",
    ),
    (
        "chipping_sparrow-06",
        "chipping_sparrow",
        40457652,
        26076708,
        "andywilson",
        156894,
        "b70edd35e00fe672a39d5f0441ea4e9bb9fac19b0f2415ae13e22160bc6091a6",
    ),
    (
        "chipping_sparrow-07",
        "chipping_sparrow",
        84151914,
        52921135,
        "davidfbird",
        145714,
        "ea65ce5ed881ded9a8157b5756fa907948f41e0eefb6735b48a1c43993e977f9",
    ),
    (
        "chipping_sparrow-08",
        "chipping_sparrow",
        523674610,
        291074747,
        "rwp84",
        47983,
        "41eac0a5fb578b089f7524c4d2e6cb38c5508aa81815d6cfc46948c081daa800",
    ),
    (
        "chipping_sparrow-09",
        "chipping_sparrow",
        292695018,
        168861389,
        "tim_kirsten",
        64812,
        "cbb2a03f5dbd2209d56f1cca8b49f342dfbaf44e51fe1014ead75b2018c6678d",
    ),
    (
        "chipping_sparrow-10",
        "chipping_sparrow",
        478480946,
        266314208,
        "russnamitz",
        61359,
        "8200cc2ec779d47be1b5afa261d934910a251a39f5620d3dc705106fb2bf5e0b",
    ),
    (
        "chipping_sparrow-11",
        "chipping_sparrow",
        220257388,
        129611350,
        "gcart043",
        129377,
        "1bdce32e7ac58321dd6beacc084afaf45973469215d66b61a0f7c612da3db361",
    ),
    (
        "chipping_sparrow-12",
        "chipping_sparrow",
        92501489,
        57964052,
        "tniernberger",
        78380,
        "44ddb9923026a97ee77ed362bc944c3d2cbbe4fbd4636000880e81613634e6c5",
    ),
    (
        "chipping_sparrow-13",
        "chipping_sparrow",
        300376697,
        172991803,
        "matthias55",
        81986,
        "7a9a5cbfb6a7d0581c98c75b5fdeb53a049d60f352eda43a39b1f9c2a3347f3b",
    ),
    (
        "chipping_sparrow-14",
        "chipping_sparrow",
        58192408,
        36778771,
        "bradenjudson",
        22352,
        "7c630d7a5b24d94e677e563a8ecb36f57fb29babe6ac1568f11537f10a5edf75",
    ),
    (
        "chipping_sparrow-15",
        "chipping_sparrow",
        80410971,
        50636049,
        "radrat",
        55646,
        "0c71ca735502e8c94db81302ecd006428206eb16d75472eb0c706e62e2656667",
    ),
    (
        "chipping_sparrow-16",
        "chipping_sparrow",
        538480223,
        298914072,
        "hiltonward",
        201271,
        "2a6479556f14a20a8c9a69ff0c1026fe4deb376944234d1d4429c558042c4346",
    ),
    (
        "chipping_sparrow-17",
        "chipping_sparrow",
        370917657,
        209626382,
        "craigmartin",
        101736,
        "feb94d319fe5b12c01e63a80dc8e44c45e15a0bfb6534ac5d2c0641205dfe1d5",
    ),
    (
        "chipping_sparrow-18",
        "chipping_sparrow",
        264119989,
        152960401,
        "laurelthrone",
        194107,
        "13e7336628f9ad784757dca82557c79eda22fdce561e0b52a10a503e72979b8e",
    ),
    (
        "chipping_sparrow-19",
        "chipping_sparrow",
        220323755,
        129631264,
        "enspring",
        85687,
        "886b7e6faa39943f0c9754e4aef637e2a2f5ad57fe8e8c7ec08360ea95af45b1",
    ),
    (
        "chipping_sparrow-20",
        "chipping_sparrow",
        210319996,
        124105091,
        "bunnymom20",
        188328,
        "2f6b6f7ac5ed9d91a361c8fed102484f4fca599ec39c9a2bfe9af1df53b28899",
    ),
    (
        "chipping_sparrow-21",
        "chipping_sparrow",
        99819947,
        62311494,
        "andy71",
        350329,
        "d649c3c9fd6b0b159a979572b48daba39fc5608104f21c6d88c7d10fa2479f7e",
    ),
    (
        "chipping_sparrow-22",
        "chipping_sparrow",
        480169680,
        267207764,
        "cvharris",
        144339,
        "d1ab8643c48f2ea823f9b61843016795def21b290c35ce3d6045e4933b5dfa0a",
    ),
    (
        "chipping_sparrow-23",
        "chipping_sparrow",
        323624945,
        185313750,
        "mrspteranodon",
        232126,
        "dd60e466086350ce1b9f7a9ba7784fc3963b3f996326ccf52f5adffa5719c39d",
    ),
    (
        "chipping_sparrow-24",
        "chipping_sparrow",
        343116928,
        195090930,
        "umamimomma",
        52983,
        "a8948a0189d19b3d7b8df65271f4844b14bd5118f510d0b9ef458c942ac41ce2",
    ),
    (
        "chipping_sparrow-25",
        "chipping_sparrow",
        354617430,
        200940828,
        "wafflemaster135",
        58919,
        "2ba9557d06a7f1bcb1c20908efc82ea6317e5a0bd0c858898f3b5c0a007d20fb",
    ),
    (
        "chipping_sparrow-26",
        "chipping_sparrow",
        352953833,
        200088501,
        "aster-asti",
        105786,
        "82c35e04e4e22fe35c9b364bce7737fa5226cb5c469c761adfc61ef0956da9db",
    ),
    (
        "chipping_sparrow-27",
        "chipping_sparrow",
        465554743,
        259357260,
        "perrydise_koisplash",
        171532,
        "35c6469fad62f198c060e056bd298f87976b39055cf66601a265ab55d5562642",
    ),
    (
        "chipping_sparrow-28",
        "chipping_sparrow",
        148831027,
        90084486,
        "ian-wolfe",
        181942,
        "cfab51f0c0598120f5312b354ae249bafb15124dd74723a8c29cdd7319206e49",
    ),
    (
        "chipping_sparrow-29",
        "chipping_sparrow",
        192945976,
        114256514,
        "sooji",
        200440,
        "04fd7d05d5ced3073d4c7ce8a4f659c6994bb485c5253b26fbc471452b4a0b14",
    ),
    (
        "white_throated_sparrow-00",
        "white_throated_sparrow",
        339621218,
        193338380,
        "rawcomposition",
        31152,
        "d1c08bfaca721bf0873437455b4cc010c6860d08b4777136c775007b0b9d07b6",
    ),
    (
        "white_throated_sparrow-01",
        "white_throated_sparrow",
        166821399,
        99992799,
        "dziakj1",
        125954,
        "c595a41fbc8948b0d918b59117340dc320e2ba80d29e92bb9dacaaed5b404852",
    ),
    (
        "white_throated_sparrow-02",
        "white_throated_sparrow",
        469820434,
        261505977,
        "joy4birds",
        111767,
        "7ce091492c73c68395667bb45578dfff11457511b0958d1d0d320d1df3e55ceb",
    ),
    (
        "white_throated_sparrow-03",
        "white_throated_sparrow",
        99351488,
        62040646,
        "bradenjudson",
        23399,
        "e103968e2a6c9efb6f0bcb548a6457aef950a4a720838a624b6796b90411538e",
    ),
    (
        "white_throated_sparrow-04",
        "white_throated_sparrow",
        177497964,
        105746665,
        "andywilson",
        29827,
        "65475d4842396f2488167453192d4aa834d2f1b40bcc78b420878c55ccf694d7",
    ),
    (
        "white_throated_sparrow-05",
        "white_throated_sparrow",
        628148203,
        344731686,
        "lavenderdame",
        106872,
        "36379abf3af51d865ba6e6804ba3dd48fc9efd12ecc0b7d97e03fc04a16178a8",
    ),
    (
        "white_throated_sparrow-06",
        "white_throated_sparrow",
        104660609,
        65043951,
        "allan7",
        42443,
        "10791891945e83a0c908c14fea07a257a0f25f51c0e708bee35184213833a635",
    ),
    (
        "white_throated_sparrow-07",
        "white_throated_sparrow",
        250718938,
        145903421,
        "stevestevens",
        138735,
        "bd52e6d2f48c247f72db0fa393ec4fa13af8510c95f0ca9134cbef71caddb6d4",
    ),
    (
        "white_throated_sparrow-08",
        "white_throated_sparrow",
        15105971,
        10793852,
        "schylerbrown",
        31467,
        "6496e7e6d3abf13b1a538f769e3cc402280cf1e9a2a1ebdf81c68dcfd7ed01e2",
    ),
    (
        "white_throated_sparrow-09",
        "white_throated_sparrow",
        171460784,
        102554447,
        "w_mark_c",
        251287,
        "3828c41af9209b408fe0d8ec6541edf35d13fc730b74fd27a82980d4f3771137",
    ),
    (
        "white_throated_sparrow-10",
        "white_throated_sparrow",
        244260317,
        142471526,
        "deejay",
        75694,
        "d67efb1ec61f6700b8a6c6552e2da9cd981e68ae81af16d6f1e0ef17fcaff84f",
    ),
    (
        "white_throated_sparrow-11",
        "white_throated_sparrow",
        194732675,
        115373159,
        "wildreturn",
        128989,
        "7ed4bde480b35734576bb4c5f9d1453e77c1f095380432b43b1b682050255831",
    ),
    (
        "white_throated_sparrow-12",
        "white_throated_sparrow",
        341990462,
        194541980,
        "ethologist",
        149260,
        "3e2710fac082cc347e6cd114d42d39f25ec47b82c25936922232eba916808cdb",
    ),
    (
        "white_throated_sparrow-13",
        "white_throated_sparrow",
        267625208,
        154862375,
        "laurelthrone",
        148616,
        "0cac1bc7053c891ce5ae1c33e3f94b69f1b19b958bd08c686db2ccaa7ba0fbd1",
    ),
    (
        "white_throated_sparrow-14",
        "white_throated_sparrow",
        330539586,
        188793537,
        "efalquet",
        119214,
        "c3db4583124472b28dbfb4c6fd9b3829e451d508a6b0a095eccc28f17e7982b4",
    ),
    (
        "white_throated_sparrow-15",
        "white_throated_sparrow",
        193091403,
        114346779,
        "ian-wolfe",
        73751,
        "e28974c8782fccb00f5ea5420140e01248ead859f151563a95a26bf67acba479",
    ),
    (
        "white_throated_sparrow-16",
        "white_throated_sparrow",
        691050773,
        377873006,
        "dinomariobob",
        167744,
        "cc5e25afa5041ce2d6ea4e0d726793843f3a867f30b8d9ccf55892d6617da887",
    ),
    (
        "white_throated_sparrow-17",
        "white_throated_sparrow",
        440986574,
        246671481,
        "suzannehale",
        141857,
        "4a354c189225de2e7b4e94a6df9cbd3a671dac0c8a7d2d4c75e933f93d8b83bf",
    ),
    (
        "white_throated_sparrow-18",
        "white_throated_sparrow",
        113217820,
        69733707,
        "kemper",
        97802,
        "4de7da3ac53a086f2c343555be5a2b62a683715db9f80636a76673cc380fa7f6",
    ),
    (
        "white_throated_sparrow-19",
        "white_throated_sparrow",
        575307217,
        318327478,
        "k-simpkins",
        90416,
        "af2b4e72a098bd90b920c8a49632e1bbff18b73954d8ac541ca81ed4baf0530b",
    ),
    (
        "white_throated_sparrow-20",
        "white_throated_sparrow",
        340420076,
        193765555,
        "don54",
        62714,
        "7f6eea434fc700166af1d343951d15f3a9c83eff06cfb0518c3c4291019a5556",
    ),
    (
        "white_throated_sparrow-21",
        "white_throated_sparrow",
        562344160,
        311510652,
        "rrfc",
        45842,
        "f7c6394649765db6e3313a03ae013291db1ae20d89d3b8f0bdf6641757f31ed0",
    ),
    (
        "white_throated_sparrow-22",
        "white_throated_sparrow",
        74598266,
        47039878,
        "ianrwhyte",
        141502,
        "b0633999572a4499a80310955ab928a86f4fe08774e669b4c4949cd26254428b",
    ),
    (
        "white_throated_sparrow-23",
        "white_throated_sparrow",
        588001234,
        324825124,
        "portablecity",
        370187,
        "11e3a97f43de5d61266d15028fe9023eef3900cfea2d027bd94ad847ecba9607",
    ),
    (
        "white_throated_sparrow-24",
        "white_throated_sparrow",
        694103481,
        379467387,
        "memoosborne",
        120656,
        "82684f82f62f6e036e206eec315fff8e1ba36d308b667eba06750b75d1fada3a",
    ),
    (
        "white_throated_sparrow-25",
        "white_throated_sparrow",
        655749108,
        359408087,
        "russnamitz",
        58449,
        "2a7d2b4226f1941d0950adc04bfd8fe62345f1138c521d0f5f0adf8bc5838da1",
    ),
    (
        "white_throated_sparrow-26",
        "white_throated_sparrow",
        599110899,
        330395670,
        "jd_flores",
        121343,
        "5eeef4f5533a4b0a0222dcf7be000a2b34f3331f3be2b939703155ba8c8ba3ae",
    ),
    (
        "white_throated_sparrow-27",
        "white_throated_sparrow",
        653888173,
        358443775,
        "toknowtheland",
        144197,
        "20a4933b00967fe4488296b2b6e89c12ecbcca0e40da591bd95927cca46ae2f7",
    ),
    (
        "white_throated_sparrow-28",
        "white_throated_sparrow",
        295037357,
        170132018,
        "sturuss",
        117077,
        "d309f75ad5f4008c906ba7b3d6138aa017f919a0a17a1d5b431fa7ab47d5f2d3",
    ),
    (
        "white_throated_sparrow-29",
        "white_throated_sparrow",
        405042885,
        228270930,
        "carterdorscht",
        122157,
        "f1fb32bcfb78f66f50ccd20f2418832afe72c7a5847bdeb8e4dc19546455cfaf",
    ),
    (
        "dark_eyed_junco-00",
        "dark_eyed_junco",
        172110799,
        102901486,
        "schylerbrown",
        182973,
        "185209c7a1111fc626a068e136ec3cfcdff3174d15f1c60af209d7b32fe7bef9",
    ),
    (
        "dark_eyed_junco-01",
        "dark_eyed_junco",
        46691943,
        29901256,
        "haida_gwaii",
        46823,
        "a92dca21e6e58c375fc313f0d1da2c86c11408beb0df8fd96fc60ae035ae80d5",
    ),
    (
        "dark_eyed_junco-02",
        "dark_eyed_junco",
        707222551,
        386266764,
        "ben142",
        289608,
        "7bf320edf4d34a4848f3e9d175cfafa66cc5bab6a1a8f97c41c670263de3ff89",
    ),
    (
        "dark_eyed_junco-03",
        "dark_eyed_junco",
        192557376,
        114006980,
        "k-simpkins",
        172398,
        "206f012add8a5d0fa434e07c51f1e7bd73a60c4d299a2202163fc662e1b03479",
    ),
    (
        "dark_eyed_junco-04",
        "dark_eyed_junco",
        8793471,
        6892999,
        "truthseqr",
        45302,
        "f8241eab39797c4e097a1432b13658466287d61ed515448457907c17e60cf5e2",
    ),
    (
        "dark_eyed_junco-05",
        "dark_eyed_junco",
        346777340,
        196961623,
        "zacharyfoster",
        46712,
        "ea8f9f0eebb4d2f86193c705344a8fab09bf634bc2217a0874ee57c4f0f5b4ab",
    ),
    (
        "dark_eyed_junco-06",
        "dark_eyed_junco",
        274980085,
        159160633,
        "andy71",
        51209,
        "c54b45ca7fdc635bdb31eb89166c9fce84f0b0f8f0c17331fd5e42b582633c9b",
    ),
    (
        "dark_eyed_junco-07",
        "dark_eyed_junco",
        243303909,
        141959574,
        "andywilson",
        71321,
        "ee5308b6f93fb40a6d795a7d8ca6f2ef55b4844513f4268ef34827e1c5e4c4a6",
    ),
    (
        "dark_eyed_junco-08",
        "dark_eyed_junco",
        213798376,
        126031618,
        "nathanael15",
        57646,
        "d6b9e7d2dcc63cdd88b47cce328149aa9e8eb486ee2a5fc0f08b90601f2c7d9b",
    ),
    (
        "dark_eyed_junco-09",
        "dark_eyed_junco",
        63482066,
        39977347,
        "chrisleearm",
        41870,
        "90573e814dbde965c70a934d9702110c40670aa22cc9f80ff8849db877d1fd72",
    ),
    (
        "dark_eyed_junco-10",
        "dark_eyed_junco",
        458710472,
        255914048,
        "joy4birds",
        90973,
        "fd25ecb1bf86896b84e9e4e8329c742011011e6883631f47dbe3744ba7463153",
    ),
    (
        "dark_eyed_junco-11",
        "dark_eyed_junco",
        20784016,
        14046286,
        "gambolingquail",
        93957,
        "a454d6f987153317c9c57c05019b08ab0ebebc46d3cf72356b014231b69a1e9d",
    ),
    (
        "dark_eyed_junco-12",
        "dark_eyed_junco",
        332842159,
        189933284,
        "jan-konilu",
        110145,
        "d18706f536164a72a4ec9bacf47437edd89d6b547f86eefdf10ed0166cebf0dd",
    ),
    (
        "dark_eyed_junco-13",
        "dark_eyed_junco",
        513004508,
        270404136,
        "thevertebratepokedex",
        141252,
        "854f75e7527932d0409e14756725388de9913c0b450536da717e54d683de169f",
    ),
    (
        "dark_eyed_junco-14",
        "dark_eyed_junco",
        593499256,
        327583635,
        "orionid",
        108583,
        "7eab4cea6d46faa878732d82c5c2ece63ecc3e66781ea08d0bd950eee581d16d",
    ),
    (
        "dark_eyed_junco-15",
        "dark_eyed_junco",
        63590391,
        40041059,
        "bobbyblackmore",
        82853,
        "87746be0bca30fc40ddba739c1a35ed1fdd41882ebfe3561518b69d285f5ae5d",
    ),
    (
        "dark_eyed_junco-16",
        "dark_eyed_junco",
        12580989,
        9282523,
        "artemis224",
        216253,
        "15198c123c01fa0f8c03edc086408c8d95ffa553083c380e8ae236dbb7056093",
    ),
    (
        "dark_eyed_junco-17",
        "dark_eyed_junco",
        12533281,
        9255403,
        "braincellsgone",
        40857,
        "be34c51655f59612858d888b05a4f927da65e1aede850c9a5f1be68fa00bfcbe",
    ),
    (
        "dark_eyed_junco-18",
        "dark_eyed_junco",
        256948665,
        149140989,
        "igor322",
        86925,
        "9339b8cf4aa347b4eb176ef1209dfaa4fcb11e277889cc17de399d6af2c8633a",
    ),
    (
        "dark_eyed_junco-19",
        "dark_eyed_junco",
        106718005,
        66230973,
        "vicki936",
        41475,
        "e6903e9a69e987c45edd468ace0bf71adf8252063ff5d39c2130409d8fea68be",
    ),
    (
        "dark_eyed_junco-20",
        "dark_eyed_junco",
        611049594,
        336277421,
        "skylar_schell",
        15425,
        "7e876479febb3d44b1e494a25f778efe4e8bfd323279031cb08f9f6fe95f742e",
    ),
    (
        "dark_eyed_junco-21",
        "dark_eyed_junco",
        591147962,
        326403963,
        "toknowtheland",
        106373,
        "07770f3313dc969802355fa8b3e62a86edb115a64771bd38938e1fab33220c39",
    ),
    (
        "dark_eyed_junco-22",
        "dark_eyed_junco",
        469746672,
        261469406,
        "shannon_j",
        74167,
        "ec4c7ce15aa1c4eadd3bc5d46d433baf9560d6a0e4593d8d57aab8fa9bec9e18",
    ),
    (
        "dark_eyed_junco-23",
        "dark_eyed_junco",
        459696953,
        256398190,
        "w_mark_c",
        262126,
        "5025e757c026e02160ff3f0e82f1f4434b888a6c9689e7ac76274c1fcec64f0c",
    ),
    (
        "dark_eyed_junco-24",
        "dark_eyed_junco",
        484171775,
        269292238,
        "aschuman",
        59459,
        "89dd933d12cf21c0e73eb01e96a9279e2e57b36676a72a9c860281d0d7703282",
    ),
    (
        "dark_eyed_junco-25",
        "dark_eyed_junco",
        357382685,
        202362003,
        "dougbrown",
        56324,
        "70275d8e07192c99e121b67d206de6823b194f043e65b9766f1f9ce772953c78",
    ),
    (
        "dark_eyed_junco-26",
        "dark_eyed_junco",
        7660371,
        6119391,
        "jeffreyleeisanaturalist",
        108394,
        "0f667e9af8e8d3585807da58a2f315c5dbcb7eece5f6862006fb9da68a4bef42",
    ),
    (
        "dark_eyed_junco-27",
        "dark_eyed_junco",
        437957476,
        245430473,
        "eug302",
        106520,
        "394abfbcf192ecf5a76cb2f19dfa2a62070fa3874b084d1468219d6ee5db6cbf",
    ),
    (
        "dark_eyed_junco-28",
        "dark_eyed_junco",
        339439778,
        193239701,
        "rawcomposition",
        32258,
        "73fad57ed975df01c529e77eaca9eab9b00741ea2cfc91358856aefca620a10a",
    ),
    (
        "dark_eyed_junco-29",
        "dark_eyed_junco",
        6198359,
        5055484,
        "glmory",
        108168,
        "d31767f42eaaf7f3527133fffb9a0271a9dbbe66fc036d5c694b605563d09965",
    ),
    (
        "house_finch-00",
        "house_finch",
        117990649,
        72375345,
        "kristen163",
        75945,
        "eeafad0dd2e91ecfe45c9d1f27dd0392a01bd81099549c60fd2e36a4b4342a9f",
    ),
    (
        "house_finch-01",
        "house_finch",
        389479656,
        220010434,
        "aster-asti",
        82128,
        "a8848197b4e7890e07538d492480c4275b75d04e10c1ae95aee91aaafe3319c5",
    ),
    (
        "house_finch-02",
        "house_finch",
        176982307,
        105476125,
        "vicki936",
        22211,
        "c377fb361df0324c7a856d9344968886ece3b94bd67188c9325b8d2d284d3a2f",
    ),
    (
        "house_finch-03",
        "house_finch",
        697940852,
        381438133,
        "ben142",
        196735,
        "a89f8e0263fdabb404b462acaa592f5dd2ac88ee4615da444470de4a1fae82d5",
    ),
    (
        "house_finch-04",
        "house_finch",
        72470599,
        45698380,
        "henrya",
        61724,
        "1b96d37a7078e1b725b80af4b10848da58b0d0c17a70c8ac01e326c0a749ee6b",
    ),
    (
        "house_finch-05",
        "house_finch",
        98576538,
        61594129,
        "enspring",
        46784,
        "8b355426d8fe6327f202c6a9458cee1b95de445bfbf7e48a4b5a2c7d0eb78a8c",
    ),
    (
        "house_finch-06",
        "house_finch",
        80751781,
        50842166,
        "leahmfulton",
        44269,
        "6d6202de26f042d83ee6c5af550cd74e6ab10eb796eed2f78c83ac9e2368e4b7",
    ),
    (
        "house_finch-07",
        "house_finch",
        630196420,
        345777550,
        "truthseqr",
        149455,
        "abb84d1e327dd82c07cbea3dd5583c07453e69b2cc220397b101e597da81bd6c",
    ),
    (
        "house_finch-08",
        "house_finch",
        214612538,
        126483167,
        "hamiltonturner",
        124116,
        "6b7687640c4641b974865da04cf9eaf1f86b774ebc19678f2fc39e55c8648930",
    ),
    (
        "house_finch-09",
        "house_finch",
        213077180,
        125637342,
        "jnicat",
        25823,
        "e1e3baff8d0bd72339e3e49089f7f4f2c1dd383ff005e49a964a1bacc4f8ebb8",
    ),
    (
        "house_finch-10",
        "house_finch",
        500744872,
        278868417,
        "pbaff",
        149626,
        "2684cfb1fc40d7766610a5920ead0ad0c27c338ccb4fb9ed18baddda150568b0",
    ),
    (
        "house_finch-11",
        "house_finch",
        268678834,
        155431721,
        "stevestevens",
        120325,
        "5d1e8c097c214d14ef7b895bf95769ae9bc25fa799a98bbfa6f12f2f84e6af79",
    ),
    (
        "house_finch-12",
        "house_finch",
        358373136,
        202884575,
        "kcthetc1",
        52329,
        "b9929163e4fdadef26c753437ac7af3ad550aa047b15cba131c05d4d335799fe",
    ),
    (
        "house_finch-13",
        "house_finch",
        104227663,
        64793290,
        "verdantpulsar",
        101003,
        "90fe37c477bad9ed30ab119e1a7445ffc2720e548a22bb65d66b2ff7b82337d5",
    ),
    (
        "house_finch-14",
        "house_finch",
        196156834,
        116218899,
        "kgarrett",
        56801,
        "ce03d1d70a89b5e6e0088f4307f8077573c3571b91997725ca2e6c69be80e2d0",
    ),
    (
        "house_finch-15",
        "house_finch",
        454332148,
        253709711,
        "rlaortiz",
        149985,
        "cf007ef8ac57bc0eb085c9eecf9fc95eb69a29df706998de237df24f9491c61d",
    ),
    (
        "house_finch-16",
        "house_finch",
        509969159,
        283798927,
        "damienxw",
        104277,
        "56d3656be473c362f1ccd09d15e62cbcfe9137d83bef7a3312d72444921c7805",
    ),
    (
        "house_finch-17",
        "house_finch",
        665287910,
        295246528,
        "dinomariobob",
        74972,
        "5f6121c1f8dbfaccbb61c279a579c22600744eb4118c2eeef09e69c86f6e1a49",
    ),
    (
        "house_finch-18",
        "house_finch",
        168402062,
        100871757,
        "k-simpkins",
        86922,
        "209a884cf7618d0b85679ae3a72f237a6d03a5a3096f6ab1b2e5503c38c50d9d",
    ),
    (
        "house_finch-19",
        "house_finch",
        247557547,
        144258909,
        "aparrot1",
        79152,
        "115302ef807e6f4d132df584a20ba8644846f05dfe9fee27a7c0e07ec89d87b7",
    ),
    (
        "house_finch-20",
        "house_finch",
        249874672,
        145517234,
        "vijaybarve",
        115757,
        "15538eeeb228d284f49f33d0bda77626b57fdaabe90d05f4379ffc3bbd855703",
    ),
    (
        "house_finch-21",
        "house_finch",
        379972789,
        214941038,
        "dougbrown",
        29723,
        "d7fea293bd3f92aec7be52bdfe5b904793c53cfaeacf9ffd2e762702ee10ba74",
    ),
    (
        "house_finch-22",
        "house_finch",
        250140068,
        145607589,
        "matthias55",
        87579,
        "62adc5d86cbdb613779fefa84f6f74f306b6fab5bfb86320f4ef40cd2571ff6a",
    ),
    (
        "house_finch-23",
        "house_finch",
        247757548,
        144364266,
        "nana10",
        123890,
        "10b3b6c0b4273a31597050df4218899ddcbe31cdfa60d9a46421ed0bf3d26558",
    ),
    (
        "house_finch-24",
        "house_finch",
        253927599,
        147557306,
        "kerykeion",
        91321,
        "6166a9bf59e2075a1129b344a24f3fcafccb610eb984510c5b4c3f9b3abe96af",
    ),
    (
        "house_finch-25",
        "house_finch",
        250651171,
        145869416,
        "cathartic_cathartes",
        55197,
        "1929f770ec5ca766c6b88ffdbad5fd7c27df033e12fe46109fd87dc1c835fc59",
    ),
    (
        "house_finch-26",
        "house_finch",
        244648136,
        142674679,
        "michelle_lopez",
        51377,
        "9422e42758a68c343d0487d07726819424a24c8271a94aa48c75f4c1e339be77",
    ),
    (
        "house_finch-27",
        "house_finch",
        373687946,
        211229903,
        "c_dizzy",
        145810,
        "ae7289dc681f8e192886d47018ee70aa9586f08e618e57fa1fe195a5695e1779",
    ),
    (
        "house_finch-28",
        "house_finch",
        242992511,
        141794074,
        "chrisleearm",
        102901,
        "b6bd72b41f04d2b6a7855b28e2b169d5aad6663db4a0ac0d364b22ffc2512018",
    ),
    (
        "house_finch-29",
        "house_finch",
        110518705,
        68278832,
        "kemper",
        103151,
        "62ee726242d1970d9bb8536e31c06635afc92b9352d1394574acb1d9d5eb26ed",
    ),
    (
        "american_goldfinch-00",
        "american_goldfinch",
        84579952,
        53187208,
        "glennberry",
        59673,
        "72d36079e592e0a83c2f774f9073bfd4cc81253452c925d1673217ddd4b52a36",
    ),
    (
        "american_goldfinch-01",
        "american_goldfinch",
        12533322,
        9255418,
        "braincellsgone",
        55661,
        "6344e0125e74791f43ac6e07e5e1b9fbfce6d19bc62b6bb5d83b3caff9f7bcbc",
    ),
    (
        "american_goldfinch-02",
        "american_goldfinch",
        131102823,
        80016788,
        "radrat",
        98990,
        "232a944f7e3351d4916a12ef2f6d598e7b007caaa95a9b064a14c1ba3af2a6ff",
    ),
    (
        "american_goldfinch-03",
        "american_goldfinch",
        175048222,
        104466897,
        "eug302",
        44231,
        "11c723482cc75fcc3a723ac1c0818a68e2fcf74c4ca684cf60195bf0d33f274e",
    ),
    (
        "american_goldfinch-04",
        "american_goldfinch",
        68849595,
        43390778,
        "mefisher",
        154503,
        "66f07bc59bb3fdedd65a4537ebabd0cafd457826b8bf4bb633181f584a3edfd1",
    ),
    (
        "american_goldfinch-05",
        "american_goldfinch",
        431916465,
        242278180,
        "k-simpkins",
        45413,
        "d76e7adf33e3a84ebec24dde5438965e62ebec8595755453973846340e4f460d",
    ),
    (
        "american_goldfinch-06",
        "american_goldfinch",
        377136648,
        213398931,
        "nathan1177",
        66516,
        "7ad75838fdf2020a8e426e97507c7dd4355da93e6eece241128c28adbe302438",
    ),
    (
        "american_goldfinch-07",
        "american_goldfinch",
        230801384,
        135330401,
        "enspring",
        42886,
        "78aebfb9b28c3e16dd9618a0e1ae06df67bfa4b8550c0879427d96915c475fed",
    ),
    (
        "american_goldfinch-08",
        "american_goldfinch",
        660472044,
        361884286,
        "ben142",
        270599,
        "733d64cd50c61334682f0862f5c7859ded34cae5776ee0bc6594fee400dc4876",
    ),
    (
        "american_goldfinch-09",
        "american_goldfinch",
        294667312,
        169935316,
        "dande",
        163470,
        "39e7892e81eeef6af4887e61bc0998e17797688eb6394fcc8c9438391e875ee0",
    ),
    (
        "american_goldfinch-10",
        "american_goldfinch",
        143215717,
        86889530,
        "memoosborne",
        129438,
        "ef4a9a771cef9ba3c2c047eb106a6aa220236dd6aaa6aade5f4ef3a37c8abcba",
    ),
    (
        "american_goldfinch-11",
        "american_goldfinch",
        403873164,
        227647158,
        "drew_baxter",
        106009,
        "74e34c776f1b9a5d375a7dfae0e309cd42dbd133b82a1ecad4a49111ac7eed49",
    ),
    (
        "american_goldfinch-12",
        "american_goldfinch",
        481153019,
        267722510,
        "vicki936",
        255397,
        "3326ddbee3270241b681cb636466e742491f110e9841f453f5158864aadaea36",
    ),
    (
        "american_goldfinch-13",
        "american_goldfinch",
        213311122,
        125763980,
        "hickl",
        24740,
        "0c295b3761bced4215519ff24a4f734773b54be98de456a4444d0819456af7dd",
    ),
    (
        "american_goldfinch-14",
        "american_goldfinch",
        417352824,
        234736320,
        "joy4birds",
        81109,
        "357014c108519543471b94f39591667d1a67d87fd1fdc4702a3a449235d6bddd",
    ),
    (
        "american_goldfinch-15",
        "american_goldfinch",
        72130720,
        45486482,
        "dctphoto",
        178111,
        "17b2f3599e20161acc17fd63bf61e9f40488b9c4d5bfaefc8cae54de42997509",
    ),
    (
        "american_goldfinch-16",
        "american_goldfinch",
        45148898,
        28952026,
        "megachile",
        49358,
        "6bbc6ca0ac074e486c20ce4c3fad5dc863cb2e908efc2b1ef97494403a351252",
    ),
    (
        "american_goldfinch-17",
        "american_goldfinch",
        133251099,
        81250513,
        "raffib128",
        128110,
        "863c76587fe1de80a84b97c2e72f38ae1cf4fa972789e006d9b0b544d696c6ef",
    ),
    (
        "american_goldfinch-18",
        "american_goldfinch",
        155797129,
        93957328,
        "nathanael15",
        74086,
        "7059ae3bdeeb48fe949e5b70b788bd28c96aecd0fe49cb7be22db9a8697bf5a5",
    ),
    (
        "american_goldfinch-19",
        "american_goldfinch",
        11400438,
        8535447,
        "akneidel",
        26423,
        "4af74c1d04ddc7bbb7bb0e9eb2977a1daf48dd9b9477d71d69a7c4cb5d4f785b",
    ),
    (
        "american_goldfinch-20",
        "american_goldfinch",
        145464774,
        88186208,
        "wildreturn",
        68204,
        "565d2a3b6d404e9ea0c24737ebcc5a3a82057b1452b4790df5e9552e8c50e592",
    ),
    (
        "american_goldfinch-21",
        "american_goldfinch",
        55990791,
        35505213,
        "conhawn",
        84915,
        "69f768c39a2180440bdcbfc6addc5d426341e8080d4cf9ba8241d57564b3e6fb",
    ),
    (
        "american_goldfinch-22",
        "american_goldfinch",
        637247336,
        289067166,
        "dinomariobob",
        142312,
        "01c63d305fce8edcc3a494f0543154c6bd6aa52368e02be84cf6c23e95b94cdd",
    ),
    (
        "american_goldfinch-23",
        "american_goldfinch",
        460988055,
        257029007,
        "eric112",
        60314,
        "0db1b5f0e32d1faf861a437a20794fa18e80ea8966313933c145451c9319e2a9",
    ),
    (
        "american_goldfinch-24",
        "american_goldfinch",
        59072170,
        37280587,
        "bradenjudson",
        23468,
        "827d77bf9ca9cc456e867434a07b68cdceb4a20e09eb9e6b67757c80cd31ff42",
    ),
    (
        "american_goldfinch-25",
        "american_goldfinch",
        109875643,
        67928198,
        "artemis224",
        217909,
        "99e1d7d30eb19d47c7974e9ddd7efe4d06329c952a41e15d0a4b2095b747df61",
    ),
    (
        "american_goldfinch-26",
        "american_goldfinch",
        66515260,
        41915252,
        "reuvenm",
        58833,
        "90f0b687d9626fdf5d0111b794cf960cf18f3de4c8eab601fcb23c41b45c4df7",
    ),
    (
        "american_goldfinch-27",
        "american_goldfinch",
        112715850,
        69464521,
        "umamimomma",
        78717,
        "fae2edcb0901dada461a9ac6b3873d6479e8faf66bfd26a0ad47021f6bcff704",
    ),
    (
        "american_goldfinch-28",
        "american_goldfinch",
        123422249,
        75440388,
        "rachel_bosley",
        131792,
        "864be3aebb7f1c8c9ed01cdf2e16b690f11354bb37e53b23d44671082afdbcfc",
    ),
    (
        "american_goldfinch-29",
        "american_goldfinch",
        252964992,
        147051902,
        "carterdorscht",
        149077,
        "a8360d1774df42f034692863781af47fd030265df9bcc1fc938333fabc673c9e",
    ),
)
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 18, "validation": 4, "test": 8}  # per species; 6 species -> 108 / 24 / 48
MIN_RECORDS = 8
MAX_RECORDS = 20_000
MIN_CLASSES = 2
MAX_CLASSES = 100
MAX_LABEL_CHARS = 64
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_ .:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def photo_url(photo_id: int) -> str:
    return f"{CORPUS_BASE_URL}{photo_id}/medium.jpg"


def observation_url(observation_id: int) -> str:
    return f"https://www.inaturalist.org/observations/{observation_id}"


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return every pinned photo (bytes keyed by record id) from the cache or the open-data bucket."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for rid, _label, photo_id, _obs, _user, size, digest in SAMPLE_RECORDS:
        local = cache / f"{photo_id}.jpg"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = photo_url(photo_id)
            if fetcher is not None:
                data = fetcher(url)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "dimer-dinov2-tutorial/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{rid} ({photo_id}/medium.jpg): fetched {len(data)} bytes with sha256 "
                    f"{_sha256_bytes(data)[:16]}…, pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[rid] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the verified photo bytes into `{id, image, label}` records with their provenance."""
    out = []
    for rid, label, photo_id, obs_id, user, _size, _digest in SAMPLE_RECORDS:
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.open(io.BytesIO(files[rid]))
        image.load()
        out.append(
            {
                "id": rid,
                "image": image.convert("RGB"),
                "label": label,
                "scientific_name": SPECIES[label][0],
                "common_name": SPECIES[label][1],
                "inat_photo_id": photo_id,
                "inat_observation_url": observation_url(obs_id),
                "observer": user,
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified draw per species: `sizes` counts per class for train / validation / test."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), []).append(dict(record))
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        needed = sum(sizes.values())
        if len(pool) < needed:
            raise ValueError(f"{label}: only {len(pool)} records available, need {needed}")
        cursor = 0
        for name, per_class in sizes.items():
            out[name].extend(pool[cursor : cursor + per_class])
            cursor += per_class
    for name in out:
        rng.shuffle(out[name])
        out[name] = [{**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(out[name])]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image/label")
    for key in ("id", "image", "label"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image, label = record["id"], record["image"], record["label"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label_name}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label_name}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"{label_name}: image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}"
        )
    if not isinstance(label, str) or not _LABEL_RE.match(label.strip()):
        raise ValueError(
            f"{label_name}: label must be a non-empty string of at most {MAX_LABEL_CHARS} plain characters"
        )
    item = {"id": rid, "image": image.convert("RGB"), "label": label.strip()}
    for key in (
        "source_id",
        "observer",
        "inat_photo_id",
        "inat_observation_url",
        "scientific_name",
        "common_name",
    ):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a labelled-image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        counts[item["label"]] = counts.get(item["label"], 0) + 1
        checked.append(item)
    if not MIN_CLASSES <= len(counts) <= MAX_CLASSES:
        raise ValueError(f"{len(counts)} distinct labels; {MIN_CLASSES}..{MAX_CLASSES} are required")
    sides = [max(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "classes": sorted(counts),
        "label_counts": dict(sorted(counts.items())),
        "image_side": {"min": min(sides), "max": max(sides)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def class_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted label vocabulary of a dataset (the `classes` a head is trained for)."""
    names = sorted({str(r["label"]) for r in records})
    if len(names) < MIN_CLASSES:
        raise ValueError(f"a dataset needs at least {MIN_CLASSES} distinct labels")
    return names


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], image_digest(r["image"]), r["label"]] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def observer_overlap(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """How many observers contributed photos to more than one split (an observation, not an assertion)."""
    seen: dict[str, set[str]] = {}
    for name, records in splits.items():
        for record in records:
            if record.get("observer"):
                seen.setdefault(str(record["observer"]), set()).add(name)
    return {"observers": len(seen), "in_more_than_one_split": sum(1 for s in seen.values() if len(s) > 1)}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_label.setdefault(record["label"], []).append(record)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        n_test = max(1, round(len(pool) * test_fraction))
        n_val = round(len(pool) * val_fraction)
        splits["test"].extend(pool[:n_test])
        splits["validation"].extend(pool[n_test : n_test + n_val])
        splits["train"].extend(pool[n_test + n_val :])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `labels.csv` (columns `id`, `file`,
    `label`) beside the image files; images are decoded, never extracted to disk."""
    source = Path(path)
    if source.is_dir():
        table = (source / "labels.csv").read_text(encoding="utf-8")
        loader = lambda name: Image.open(source / name)  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "labels.csv" not in members:
            raise ValueError("BYOD zip must contain labels.csv")
        table = archive.read(members["labels.csv"]).decode("utf-8")
        loader = lambda name: Image.open(io.BytesIO(archive.read(members[name])))  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding labels.csv and the image files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"labels.csv is missing columns {sorted(missing)}")
    out = []
    for row in rows:
        image = loader(row["file"])
        image.load()
        out.append({"id": row["id"], "image": image.convert("RGB"), "label": row["label"]})
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the labels table of a split (id, file, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "file", "label", "observer", "inat_observation_url"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "file": f"{record['inat_photo_id']}.jpg" if record.get("inat_photo_id") else record["id"],
                    "label": record["label"],
                    "observer": record.get("observer", ""),
                    "inat_observation_url": record.get("inat_observation_url", ""),
                }
            )
    return out
