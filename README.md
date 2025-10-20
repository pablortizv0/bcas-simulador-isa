\# Bcas ISA Calc API



API en FastAPI que calcula:

\- CAP óptimo (`cap\_final`) ajustado a una TIR objetivo.

\- CAPs de amortización anticipada en 6/12/18 meses (`ventana1/2/3`).



\## Endpoints



\### `GET /health`

Salud de la API.



\### `POST /propuesta`

\*\*Body (ejemplo mínimo):\*\*

```json

{

&nbsp; "importe\_financiado": 7000,

&nbsp; "duracion\_curso": 8,

&nbsp; "categoria\_centro\_legible": "Excelente",

&nbsp; "descuento\_tramos": \[

&nbsp;   { "limite\_superior": 2000, "descuento": 0.05 },

&nbsp;   { "limite\_superior": 4000, "descuento": 0.06 },

&nbsp;   { "limite\_superior": 6000, "descuento": 0.07 },

&nbsp;   { "limite\_superior": 8000, "descuento": 0.08 },

&nbsp;   { "limite\_superior": 1e12, "descuento": 0.10 }

&nbsp; ]

}



