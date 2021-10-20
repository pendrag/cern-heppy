# coding=utf-8
#
# Lint as: python3

"""Heppy data loader."""

from xml.dom import minidom
import datasets

_CITATION = """\
@Article{montejo2004, 
    author = {Montejo-Ráez, A. and Steinberger, R. and Ureña-López, L. A.},
    title = {Adaptive selection of base classifiers in one-against-all learning for large multi-labeled collections},
    booktitle = {Advances in Natural Language Processing: 4th International Conference, EsTAL 2004},
    pages = {1--12},
    year = {2004},
    editor = {Vicedo J. L. et al.}, 
    location = {Alicante, Spain}, 
    number = {3230}, 
    series = {Lectures notes in artifial intelligence}, 
    publisher = {Springer} 
}
"""

_DESCRIPTION = """\
Este corpus está orientado al estudio de clasificadores de texto multi-etiquetado. \
Está compuesto por artículos científicos en el área de la Física de Altas Energías \
(HEP – High Energy Physics) obtenidos del servidor de documentos CDS del Laboratorio \
de Física Nuclear Europeo (CERN). El corpus está dividido en tres subconjuntos \
(denominadas particiones), donde cada partición se compone, a su vez, de dos ficheros: \
uno que contiene los registros de cada artículo (con información como los abstract, los \
autores y, por supuesto, las clases o palabras clave) en formato XML comprimido, y otro \
que contiene una versión en texto plano del artículo completo generado a partir del PDF \
disponible en las bases de datos del CERN (en formato tar + gzip) Las clases están \
delimitadas por la marca XML KEYWORD. Estas son las etiquetas del tesauro de DESY \
asignadas manualmente. Puede obtener más información sobre el tesauro de DESY.

Partición hepth: 18,114 documentos de Física Teórica (metadatos – 5,3 Mb) (artículos – 226 Mb)
Partición hepex: 2,599 documentos de Física Experimental (metadatos – 1,6 Mb) (artículos – 28 Mb)
Partición astroph: 2,716 documentos de Astrofísica (metadatos – 1,1 Mb) (artículos – 29 Mb)
"""

_URL = "https://sinai.ujaen.es/recursos/download/hep-corpus/v3.0/"

_URLS = {
    "astroph": _URL + "metadata-astroph.xml",
    "hepex": _URL + "metadata-hepex.xml",
    "hepth": _URL + "metadata-hepth.xml"
}


class HeppyConfig(datasets.BuilderConfig):
    "BuilderConfig for Hepth-Abstract."

    def __init__(self, **kwargs):
        super(HeppyConfig, self).__init__(**kwargs)


class Heppy(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        HeppyConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "subject": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "keywords": datasets.features.Sequence(
                        datasets.Value("string")
                    )
                }
            ),

            supervised_keys=None,
            homepage="https://sinai.ujaen.es/investigacion/recursos/coleccion-hep",
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        downloaded_files = {}
        if self.config.data_dir:
            print(f"Using local files...")
            downloaded_files['astroph'] = self.config.data_dir + 'metadata-astroph.xml'
            downloaded_files['hepex'] = self.config.data_dir + 'metadata-hepex.xml'
            downloaded_files['hepth'] = self.config.data_dir + 'metadata-hepth.xml'
        else:
            print("Downloading files...")
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name='astroph', gen_kwargs={"filepath": downloaded_files['astroph']}),
            datasets.SplitGenerator(name='hepex', gen_kwargs={"filepath": downloaded_files['hepex']}),
            datasets.SplitGenerator(name='hepth', gen_kwargs={"filepath": downloaded_files['hepth']})
        ]


    def _generate_examples(self, filepath):
        """This function yields every example if every column exists."""

        file = minidom.parse(filepath)

        records = file.getElementsByTagName("record")

        for id, record in enumerate(records):
            title_el = record.getElementsByTagName("title")
            if not title_el:
                continue
            title = title_el[0].firstChild.data

            subject_el = record.getElementsByTagName("subject")
            if not subject_el:
                continue
            subject = subject_el[0].firstChild.data

            description_el = record.getElementsByTagName("description")
            if not description_el:
                continue
            description = description_el[0].firstChild.data

            keywords = [keyword.firstChild.data for keyword in record.getElementsByTagName("keyword")]
            if not keywords:
                continue
            
            yield id, {
                "title": title,
                "subject": subject,
                "description": description,
                "keywords": keywords
            }
