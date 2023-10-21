NONE_LABEL = 'X' # must not be equal to any label abbreviation in label definition dict

LABEL_DEF = {
    'OBJETIVOS': 'O', 'AGENTE': 'A', 'FACILITADORES': 'F',
    'PARTIDARIOS': 'P', 'VÍCTIMAS': 'V', 'EFECTOS_NEGATIVOS': 'E',
}

# fields that hold binary labeling information in json 'data' field
CONSPI_FIELDS = ['GS_CONSPI_THEORY', 'GS_CONSPI-THEORY']
CT_FIELDS = ['GS_CRITICAL_THINKING']
