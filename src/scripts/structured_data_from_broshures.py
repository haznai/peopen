from typing_extensions import Literal
from pydantic import BaseModel
from openai import OpenAI

class Argument(BaseModel):
    ueberschrift: str
    inhalt: str

class ImDetail(BaseModel):
    ueberschrift: str
    inhalt: str

class Volksiniative(BaseModel):
    titel: str
    im_detail: list[ImDetail]
    argumenteKomitee: list[Argument]
    empfehlungKomitee: Literal["ja", "nein"]
    argumenteBundesrat: list[Argument]
    empfehlungKomitee: Literal["ja", "nein"]

class StructuredData(BaseModel):
    volksiniativen: list[Volksiniative]

client = OpenAI()

# todo: give examples
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "system", "content": r"""
            Du bist ein hilfreicher Chatbot der halb-strukturierte OCR-Texte in JSON-Strukturen umwandelt.
            ch gebe dir den Text einer schweizerischen Abstimmungsbroschüre. Der Text wurde mit OCR extrahiert und ist somit nicht in einem ganz richtigen Format. Die Abstimmungsbroschüre beschreibt Schweizer Volksinitiativen. Für jede beschriebene Initiative liefert die Broschüre detaillierte Informationen für die Volksinitiativen, Argumente seitens des Initiativkomitees, des Bundesrates und Parlaments.

            Aufgrund des OCR-Verfahrens ist der Text nicht verlässlich strukturiert. Bitte hilf mir, dem Text eine JSON-Struktur zu geben. Die Struktur soll wie folgt aussehen (Musterinhalt):

            ```json
            {
	"volksiniativen": [
		// Beispiel Volksiniative 1
		{
			"titel": "Prämien-Entlastungs-Initiative",
			// "Im Detail" info über die Iniative
			"im_detail": [
				{
					// Jeder Paragraph hat eine überschrift im text
					"ueberschrift": "Ausgangslage",
					"inhalt": "Wer in der Schweiz krank ist, erhält die nötige medizinische Behandlung. Seit 1996 übernimmt die obligatorische Krankenversicherung die Kosten dafür. Die Krankenversicherung wird über die Krankenkassenprämien und Kostenbeteiligungen (Franchise, Selbstbehalt, Spitalkostenbeitrag) finanziert. Die Kosten der Krankenversicherung sind in den letzten Jahrzehnten stark gestiegen. Um sie zu decken, mussten die Prämien entsprechend erhöht werden. Die Prämien stiegen im Verhältnis deutlich mehr als die Löhne"
				},
				{
					"ueberschrift": "Prämienverbilligung",
					"inhalt": "Die Prämien werden pro Person und unabhängig von derEinkommenshöhe bestimmt. Die Kantone sind verpflichtet, die Prämien der Versicherten in bescheidenen wirtschaftlichen Verhältnissen zu verbilligen. Die Kantone erhalten dazu vomBund einen Beitrag. Der Mittelstand profitiert jedoch nicht oder nur teilweise von dieser Verbilligung und wird darum von den steigenden Prämien zunehmend stark belastet"
				},
				...
			]
			// Argumentenliste des Komitees
			"argumenteKomitee": [
				{
					// jedes argument hat eine überschrift im text
					"ueberschrift": "Worum geht es?",
					// das argument
					"inhalt": "Die Krankenkassenprämien steigen seit Jahren und reissen ein immer grösseres Loch in unser Portemonnaie. Bis zu 15 000 Franken: So viel zahlt heute eine vierköpfige Familie pro Jahr für die Krankenkasse. Die Prämienexplosion ist aber nur ein Spiegelbild der steigenden Kosten im Gesundheitswesen. Um das Problem nachhaltig zu lösen, braucht es jetzt die Kostenbremse."
				},
				{
					"ueberschrift": "Drohen

            Rationierungen?",
					"inhalt": "Nein. Im Gegenteil: Die Initiative will, dass alle Akteur eendlich Verantwortung für die Kostenexplosion übernehmen und der interne Verteilkampf zulasten der Prämienzahlenden
            aufhört. Während Hausärztinnen, Kinderärzte und Pflegende
            schon heute die Lasten des Systems tragen, bereichern sich
            andere schamlos."

				},
				...
			]
			// die broschüre enthält die Info ob das komitee eine Ja oder Nein stimme empfiehlt
			"empfehlungKomitee": "ja",
						// Argumentenliste des Bundesrates und Parlaments
			"argumenteBundesrat": [
				{
					// jedes argument hat eine überschrift im text
					"ueberschrift": "Richtige Diagnose, falsches Mittel",
					// das argument
					"inhalt": "Die Initiative greift ein wichtiges Problem auf: Die Kosten in der obligatorischen Krankenversicherung steigen zu stark. Es gibt ineffiziente Strukturen und es werden mehr Behandlungen durchgeführt, als medizinisch nötig wären. Die Initiative ist
            aber zu starr: Sie bindet das erlaubte Kostenwachstum einseitig
            an die Entwicklung der Löhne und der Wirtschaft. Damit
            werden nachvollziehbare Gründe für das Kostenwachstum
            ausgeblendet, beispielsweise der medizinische Fortschritt oder die Alterung der Bevölkerung."
				},
				...
			]
			// die broschüre enthält die Info ob derBunesrat  eine Ja oder Nein stimme empfiehlt
			"empfehlungBundesrat": "nein"
		},
		{
			"titel": "Kostenbremse-Initiative"
			...
		}
            }
            ```

            Es interessieren nur die Argumente, alle anderen Informationen können ignoriert werden.
            """},
        {"role": "user", "content": r"""
            ---------- OCR-Text von der Broschüre
            Im Detail **Volksinitiative «Für ein besseres**
            Leben im Alter (Initiative für eine 13. AHV-Rente)»

            | Argumente Initiativkomitee        | 14   |
            |-----------------------------------|------|
            | Argumente Bundesrat und Parlament | 16   |
            | Abstimmungstext                   | 18   |

            | Der Auftrag  der AHV                                                                                                                                                                                                                                                                                                                                                                                                                                      | Die Alters- und Hinterlassenenversicherung (AHV) ist das    |
            |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
            | wichtigste Sozialwerk der Schweiz: Alle Menschen in der  Schweiz haben im Alter Anspruch auf eine Rente der AHV. Die  Verfassung legt fest, dass die AHV-Renten den Existenzbedarf  angemessen decken müssen. Die Mehrheit der Pensionierten  bestreitet ihren Lebensunterhalt mit zusätzlichen Einkünften,  insbesondere mit Renten aus der Pensionskasse. Wer den  Existenzbedarf damit nicht decken kann, hat Anspruch auf  Ergänzungsleistungen (EL). |                                                             |
            | Die Altersrenten  der AHV                                                                                                                                                                                                                                                                                                                                                                                                                                 | Die minimale ganze AHV-Altersrente beträgt zurzeit          |
            | 1225 Franken, die maximale Rente 2450 Franken pro Monat.1 Ehepaare und Paare in eingetragener Partnerschaft erhalten  zusammen höchstens das Anderthalbfache einer maximalen  Einzelrente, also 3675 Franken. Übersteigen die beiden Renten  diesen Betrag, werden sie gekürzt. Der Bundesrat passt alle  AHV-Renten regelmässig an die Preis- und Lohnentwicklung  an. Die letzte Anpassung erfolgte auf Anfang 2023.                                    |                                                             |
            | Initiative will eine  13. AHV-Rente                                                                                                                                                                                                                                                                                                                                                                                                                       | Die Initiative fordert, dass die monatliche Rente nicht nur |
            | 12 Mal, sondern 13 Mal pro Jahr ausbezahlt wird. Dies in Anlehnung an den 13. Monatslohn, den viele Arbeitnehmerinnen  und Arbeitnehmer erhalten. Dies entspricht einer Erhöhung  der jährlichen AHV-Rente um 8,3 Prozent. Die minimale jährliche  Altersrente würde von 14 700 auf 15925 Franken, die maximale  Altersrente von 29 400 auf 31 850 Franken steigen. Paare  hätten statt 44100 Franken maximal 47775 Franken zugut.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                             |

            1 Die Höhe der AHV-Rente ist abhängig vom durchschnittlichen Einkommen während der Beitragspflicht sowie von Erziehungsund Betreuungsgutschriften. Personen, die Beitragslücken haben, also nicht jedes Jahr in die AHV einbezahlt haben, erhalten nicht eine ganze AHV-Rente, sondern eine Teilrente.

            Jährliche AHV-Rente heute und nach Annahme der Initiative Bei Annahme der Initiative käme zu den 12 Monatsrenten jedes Jahr eine 13. Monatsrente dazu. CHF
            50000

            ![2_image_0.png](2_image_0.png)

            12 Monatsrenten
            - 13. AHV-Rente
            *Ehepaare und Personen in eingetragener Partnerschaft erhalten zusammen höchstens das Anderthalbfache einer maximalen ganzen Rente. Quelle: Bundesamt für Sozialversicherungen BSV
            Ergänzungsleistungen bleiben trotz 13. Rente erhalten Pensionierte, die den Existenzbedarf nicht decken können, haben Anspruch auf Ergänzungsleistungen. Das sind insbesondere Personen in Pflegeheimen, welche die hohen Heimkosten nicht selber tragen können. Häufig sind es auch Pensionierte, die nur eine AHV-Rente, aber kein oder wenig Vermögen haben. Steigen ihre Einnahmen, kann das dazu führen, dass die Ergänzungsleistungen entsprechend gesenkt oder gar gestrichen werden. Die Initiative bestimmt, dass diese Regel bei der 13. AHV-Rente nicht angewendet wird. Somit bekämen alle Pensionierten mehr Geld, auch diejenigen mit Ergänzungsleistungen.

            | Andere Renten  bleiben gleich             | Die AHV bezahlt nicht nur Altersrenten, sondern auch   |
            |-------------------------------------------|--------------------------------------------------------|
            | Hinterlassenenrenten an Witwen, Witwer und Waisen. Zusätzlich sorgt die Invalidenversicherung (IV) für die Existenzsicherung von Personen mit gesundheitlichen Einschränkungen. Alle  diese Leistungen der 1. Säule sind aufeinander abgestimmt.  Mit der Initiative würden nur die Altersrenten der AHV erhöht,  die anderen Renten hingegen weiterhin 12 Mal pro Jahr  bezahlt.                                           |                                                        |
            | Finanzielle  Auswirkungen  der Initiative | Die jährlichen Ausgaben der AHV betragen heute rund    |
            | 50 Milliarden Franken. Die 13. AHV-Rente würde bei der Einführung voraussichtlich etwa 4,1 Milliarden Franken zusätzlich  kosten.2  Davon müsste der Bund rund 800 Millionen Franken  übernehmen. Die zusätzlichen Kosten für die 13. AHV-Rente  würden Jahr für Jahr ansteigen, weil die Zahl der Rentnerinnen  und Rentner stark wächst. Fünf Jahre nach Einführung würden  die Kosten voraussichtlich rund 5 Milliarden Franken betragen.                                           |                                                        |
            | Finanzierung offen                        | Die Initiative macht keine Angaben dazu, wie die zusätz                                                        |
            | lichen Ausgaben für die 13. AHV-Rente finanziert werden sollen.  Das müsste vom Parlament bestimmt werden. Heute wird  die AHV hauptsächlich mit Lohnbeiträgen, mit dem Beitrag des  Bundes und mit Einnahmen aus der Mehrwertsteuer gespeist.  Würden die zu erwartenden zusätzlichen Ausgaben der AHV  für die 13. Rente bei deren Einführung über die Lohnbeiträge  finanziert, müssten diese von 8,7 auf 9,4 Prozent erhöht werden. Diese Erhöhung ginge je zur Hälfte zulasten der Arbeitnehmenden und der Arbeitgebenden. Bei einer Finanzierung  über die Mehrwertsteuer müsste diese von 8,1 auf 9,1 Prozent  angehoben werden. In Frage kämen auch andere Finanzierungsmassnahmen oder eine Kombination davon.                                           |                                                        |

            2 Finanzperspektiven der AHV des Bundesamts für Sozialversicherungen BSV ( bsv.admin.ch > Sozialversicherungen > AHV > Reformen & Revisionen > Volksinitiative «Für ein besseres Leben im Alter»)
            Massnahmen zur Stabilisierung der AHV
            In den letzten Jahren wurden verschiedene Massnahmen zur Sicherung der AHV verabschiedet. So wurden im Jahr 2020 die Lohnabzüge und der Bundesbeitrag für die AHV erhöht und auf Anfang 2024 die Mehrwertsteuersätze für die AHV angehoben, und bis 2028 wird das AHV-Alter der Frauen auf 65 heraufgesetzt. Diese Reformen hat das Volk 2019 und 2022 angenommen. Sie stabilisieren die Finanzen der AHV bis 2030. Danach ist mit Defiziten zu rechnen. Darum hat das Parlament den Bundesrat beauftragt, bis 2026 eine Reform für die Zeit nach 2030 auszuarbeiten. Diese Reform müsste die höheren Ausgaben wegen der 13. AHV-Rente mitberücksichtigen und rechtzeitig verabschiedet werden, damit die Finanzen der AHV im Gleichgewicht bleiben.

            13

            | Argumente                                                                                                                                                                                                                                                                                                                                                                                                 | Initiativkomitee Mieten, Krankenkassenprämien, Lebensmittel: Alles ist  teurer. Die Rente reicht immer weniger weit. Wer ein Leben  lang gearbeitet und in die Altersvorsorge einbezahlt hat,  verdient eine anständige Rente. Deshalb braucht es nun eine  13. AHV-Rente. Sie verbessert die Situation der heutigen und  zukünftigen Rentnerinnen und Rentner rasch und effizient.  Deshalb: Ja zur 13. AHV-Rente.   |
            |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
            | Darum geht es                                                                                                                                                                                                                                                                                                                                                                                             | Das Stimmvolk hat die AHV eingeführt, damit alle in der                                                                                                                                                                                                                                                                                                                                                               |
            | Schweiz nach einem langen Erwerbsleben anständig leben  können. Doch heute haben immer mehr Rentnerinnen und  Rentner Mühe, über die Runden zu kommen. Bei Annahme der  Initiative wird eine 13. AHV-Rente ausbezahlt, analog  zum 13. Monatslohn. Auch Bezügerinnen und Bezüger von  Ergänzungsleistungen bekommen die zusätzliche Rente.  Das gleicht schnell und effizient die gestiegenen Preise aus. |                                                                                                                                                                                                                                                                                                                                                                                                                       |
            | Die Rente reicht  nicht mehr                                                                                                                                                                                                                                                                                                                                                                              | Mieten, Krankenkassenprämien, Strom und Lebensmittel                                                                                                                                                                                                                                                                                                                                                                  |
            | sind teurer. Die höheren Lebenshaltungskosten fressen eine  Monatsrente weg. Und die Pensionskassenrenten sinken seit  Jahren. Darum braucht es rasch eine Erhöhung der Altersrenten  für aktuelle und zukünftige Rentnerinnen und Rentner.                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                       |
            | Beste Lösung für  anständige Renten                                                                                                                                                                                                                                                                                                                                                                       | Die AHV kommt allen in der Schweiz zugute. Für die                                                                                                                                                                                                                                                                                                                                                                    |
            | meisten Arbeitnehmenden lohnt sie sich: Die Arbeitgeber  tragen die Hälfte der Beiträge. Auch Topverdienende zahlen  einen Teil der Rente, weil ihre Millionen-Boni AHV-pflichtig  sind. Ausserdem hat die AHV stabile und tiefe Kosten.  Alle Erträge fliessen direkt in die Renten, ohne dass Banken,  Vermittler oder Aktionäre mitverdienen.                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                       |
            | AHV ist für Frauen  besonders wichtig                                                                                                                                                                                                                                                                                                                                                                     | Die höheren Preise treffen Menschen mit tieferer Rente                                                                                                                                                                                                                                                                                                                                                                |
            | besonders hart. Darunter sind überdurchschnittlich viele Frauen.  Eine 13. AHV-Rente bringt ihnen am meisten: Nur aus der AHV  haben alle eine Rente. Nur die AHV anerkennt die unbezahlte  Betreuungsarbeit, die hauptsächlich von Frauen geleistet wird:  Ein Kind grosszuziehen, erhöht die AHV-Rente.                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                       |

            Finanzielle Mittel sind vorhanden Die AHV verzeichnet Überschüsse. Heute hat sie mit fast 50 Milliarden Franken so hohe Reserven wie noch nie. Die 13. AHV-Rente kostet bei der Einführung rund 4,1 Milliarden. Im gleichen Jahr schreibt die AHV gemäss Bundesrat einen Überschuss von 3,5 Milliarden. Die Kosten der 13. AHVRente sind also zu einem grossen Teil schon gedeckt. Für die langfristigen Finanzierungsbedürfnisse reicht zum Beispiel ein zusätzlicher Lohnbeitrag von 0,4% der Arbeitnehmenden. Zusammen mit den Beiträgen der Arbeitgeber bringt das jährlich zusätzliche 3,7 Milliarden ein.

            Empfehlung des Initiativkomitees Darum empfiehlt das Initiativkomitee:
            Ja AHVx13.ch Der Text auf dieser Doppelseite stammt vom Initiativkomitee. Es ist für den Inhalt und die Wortwahl verantwortlich.

            ## Argumente Bundesrat Und Parlament

            | Die Initiative für eine 13. AHV-Rente hätte zusätzliche Kosten  in Milliardenhöhe zur Folge und würde die Finanzierungsprobleme der AHV erheblich verschärfen. Auch ohne 13. AHV-Rente  ist die finanzielle Stabilität der AHV mittelfristig gefährdet:  Geburtenstarke Jahrgänge erreichen das AHV-Alter und die  Lebenserwartung steigt. Eine 13. AHV-Rente ist auch gar nicht  nötig: Die grosse Mehrheit der Pensionierten ist darauf nicht  angewiesen. Bundesrat und Parlament lehnen die Vorlage  insbesondere aus folgenden Gründen ab:                                                                                                                                                                                                                                                                                                                                                                                             |                                                            |
            |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
            | Mehrkosten  belasten AHV  zu stark                                                                                                                                                                                                                                                                                                                                                          | Die Initiative würde die AHV finanziell zu stark belasten. |
            | Die Ausgaben der AHV würden auf einen Schlag um mehr als  4 Milliarden steigen und danach schnell weiter zunehmen. Die  Rechnung der AHV würde rasch aus dem Gleichgewicht geraten. Die Zusatzkosten wären für die AHV ohne substanzielle  neue Einnahmen oder kostensenkende Massnahmen wie die  Erhöhung des Rentenalters nicht zu verkraften.                                                                                                                                                                                                                                                                                                                                                                                             |                                                            |
            | Initiative  verteuert Arbeit  oder Konsum                                                                                                                                                                                                                                                                                                                                                   | Die Finanzierung der 13. Rente ginge auf Kosten der        |
            | arbeitenden Bevölkerung und der Unternehmen oder der  Konsumentinnen und Konsumenten. Um die hohen Kosten für  eine 13. AHV-Altersrente zu decken, müssten beispielsweise die  Lohnabzüge oder die Mehrwertsteuer weiter erhöht werden.  Damit würde die Arbeit verteuert oder die Preise würden steigen.                                                                                   |                                                            |
            | Höhere Steuern  oder weniger  Ausgaben                                                                                                                                                                                                                                                                                                                                                      | Eine 13. AHV-Altersrente hätte auch negative Auswirkun                                                            |
            | gen auf den Bundeshaushalt, weil der Bund rund einen Fünftel  der jährlichen Ausgaben der AHV bezahlen muss. Er hätte auf  einen Schlag Mehrkosten von mehr als 800 Millionen Franken,  die Jahr für Jahr zunehmen würden. Der Bund müsste seine  Steuern erhöhen oder Ausgaben kürzen.                                                                                                     |                                                            |
            | Hohe Kosten,  geringer sozialer  Nutzen                                                                                                                                                                                                                                                                                                                                                     | Der soziale Nutzen der 13. AHV-Rente wäre gering. Eine     |
            | grosse Mehrheit der Pensionierten erhält neben der AHV-Rente  Leistungen der Pensionskasse; viele haben zudem noch andere  Einkommen oder Vermögen. Mit der Initiative würden viele  Pensionierte eine 13. AHV-Rente erhalten, obwohl sie darauf  nicht angewiesen sind. Rentnerinnen und Rentner, die ihren  Existenzbedarf nicht decken können, haben Anspruch auf  Ergänzungsleistungen. |                                                            |

            Empfehlung von Bundesrat und Parlament

            | In den letzten fünf Jahren waren zwei schwierige Refor   |
            |---|
            | men nötig, um die AHV-Finanzen für die nächsten zehn Jahre  zu stabilisieren. Sie haben insbesondere der Bevölkerung  im erwerbsfähigen Alter zusätzliche Lasten auferlegt. Und die  nächste Reform ist bereits aufgegleist, damit die AHV auch  mittelfristig nicht aus dem Gleichgewicht gerät. Der Bundesrat  wird dem Parlament bis 2026 Vorschläge unterbreiten, wie  die Finanzen der AHV für die Zeit nach 2030 stabilisiert werden  können. Anstatt der AHV weitere Ausgaben aufzubürden,  müssen wir dafür sorgen, dass die Renten der AHV gesichert  werden. Aus all diesen Gründen empfehlen Bundesrat und Parlament, die Volksinitiative «Für ein besseres Leben im Alter  (Initiative für eine 13. AHV-Rente)» abzulehnen.   |

            Nein admin.ch/13-AHV-renten

            | Sicherung der  Renten hat  Priorität   |
            |----------------------------------------|

            Im Detail Volksinitiative «Für eine sichere und nachhaltige Altersvorsorge (Renteninitiative)»

            | Argumente Initiativkomitee        | 26   |
            |-----------------------------------|------|
            | Argumente Bundesrat und Parlament | 28   |
            | Abstimmungstext                   | 30   |

            | Finanzen der AHV  bis 2030  stabilisiert                                      | Die AHV ist das Fundament der Altersvorsorge. Die Men                                                       |
            |-------------------------------------------------------------------------------|-------------------------------------------------------|
            | schen in der Schweiz müssen sich auf ihre Renten verlassen  können. Um die Renten zu sichern, haben Bundesrat und  Parlament in den letzten fünf Jahren zwei Reformen beschlossen. Diese erhöhen die Einnahmen und senken die Ausgaben  der AHV. So wurden die Lohnbeiträge und die Mehrwertsteuer  angehoben, und ab 2025 wird das Rentenalter der Frauen  schrittweise auf 65 Jahre erhöht. Das Volk hat diesen Reformen  zugestimmt. Damit sind die Finanzen der AHV bis etwa 2030  stabilisiert.1                                                                               |                                                       |
            | Herausforderungen  der AHV                                                    | Für die Zeit nach 2030 sind weitere Massnahmen notwen                                                       |
            | dig, um die Renten zu sichern. Vor allem aus zwei Gründen:  Erstens wächst die Zahl der Rentnerinnen und Rentner schneller als die Zahl der Erwerbstätigen, welche die Renten finanzieren. Zweitens steigt die Lebenserwartung, und deswegen  müssen die Renten länger ausbezahlt werden. Das Parlament  hat den Bundesrat deshalb bereits damit beauftragt, eine  weitere Vorlage zur Stabilisierung der AHV für die Zeit nach  2030 zu erarbeiten.                                                                               |                                                       |
            | Initiative                                                                    | Die Initiative will die Finanzierung der AHV mit der Erhö                                                       |
            | hung des Rentenalters nachhaltig sichern und einen Automatismus zur Berechnung des Rentenalters in der Verfassung  verankern. Sie sieht zwei Etappen vor:                                                                               |                                                       |
            | Rentenalter 66  bis 2033                                                      | Zuerst soll das Rentenalter für Männer und Frauen auf |
            | 66 Jahre erhöht werden. Dies würde schrittweise von 2028 bis  2033 geschehen. |                                                       |

            1 Berechnungen des Bundeamtes für Sozialversicherungen BSV
            ( bsv.admin.ch > Sozialversicherungen > AHV > Reformen & Revisionen > Volksinitiative «Für eine sichere und nachhaltige Altersvorsorge»)

            | Rentenalter an  Lebenserwartung  gebunden                                                                                                                                                                                          | Nach 2033 soll das Rentenalter automatisch weiter stei                                                         |
            |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
            | gen, wenn die durchschnittliche Lebenserwartung zunimmt.  Die Erhöhung des Rentenalters soll dem Anstieg der Lebenserwartung allerdings nicht eins zu eins folgen, sondern nur zu  80 Prozent. Rein rechnerisch würde das beispielsweise bedeuten, dass bei einem Anstieg der Lebenserwartung um einen  Monat das Rentenalter um 0,8 Monate erhöht würde. Wie  dieser Automatismus genau ausgestaltet wäre, müssten Bundesrat und Parlament bei der Umsetzung der neuen Verfassungsbestimmung festlegen.                                                                                                                                                                                                                                    |                                                         |
            | Maximal zwei  Monate pro Jahr                                                                                                                                                                                                      | Die Erhöhung des Rentenalters dürfte ab 2033 nicht mehr |
            | als zwei Monate pro Jahr betragen, auch wenn der Anstieg  der Lebenserwartung eine stärkere Erhöhung erfordern würde.  Jede Erhöhung müsste den betroffenen Personen fünf Jahre  vor Erreichen des Rentenalters mitgeteilt werden. |                                                         |
            | Rentenalter 67  im Jahr 2043                                                                                                                                                                                                       | Massgeblich für die Bestimmung des Rentenalters wäre    |
            | die durchschnittliche Lebenserwartung der Schweizer Wohnbevölkerung im Alter von 65 Jahren. Derzeit können 65-Jährige  im Durchschnitt mit noch rund 22 Lebensjahren rechnen. Laut  den Bevölkerungsszenarien des Bundesamtes für Statistik ist  davon auszugehen, dass die Lebenserwartung weiter steigen  wird - und zwar um etwas mehr als einen Monat pro Jahr.  Trifft das zu, so würde das Rentenalter gemäss dem Automatismus der Initiative bis ins Jahr 2043 auf 67 Jahre ansteigen.2                                                                                                                                                                                                                                    |                                                         |

            2 «Lebenserwartung, 2000–2022», Bundesamt für Statistik BFS
            ( bfs.admin.ch > Statistiken finden > Bevölkerung > Geburten und Todesfälle > Lebenserwartung > Tabellen); Berechnungen des Bundeamtes für Sozialversicherungen BSV ( bsv.admin.ch >
            Sozialversicherungen > AHV > Reformen & Revisionen > Volksinitiative «Für eine sichere und nachhaltige Altersvorsorge»)
            Anstieg des Rentenalters bei Annahme der Initiative Ab 2028 würde das Rentenalter bis 2033 schrittweise auf 66 Jahre erhöht und danach automatisch an die Lebenserwartung angepasst.

            ![13_image_0.png](13_image_0.png)

            Die Erhöhung des Frauenrentenalters auf 65 Jahre wurde mit der Reform AHV 21 bereits beschlossen.
            Quelle: Berechnungen des Bundesamts für Sozialversicherungen BSV

            | Finanzielle  Auswirkungen  der Initiative   | Die Erhöhung des Rentenalters würde in der AHV zu   |
            |---------------------------------------------|-----------------------------------------------------|
            | höheren Einnahmen und tieferen Ausgaben führen: Weil die  Menschen länger arbeiten, bezahlen sie länger AHV-Beiträge  und beziehen erst später eine Rente. Ab dem Jahr 2033, wenn  das Rentenalter 66 erreicht wäre, würde die Rechnung der AHV  voraussichtlich um jährlich rund 2 Milliarden Franken entlastet.  Danach würde die AHV mit jeder automatischen Erhöhung des  Rentenalters zusätzlich erheblich entlastet. Die Rentenaltererhöhung allein generiert aus heutiger Sicht aber nicht genug  finanzielle Mittel zur langfristigen Sicherung der AHV-Finanzen. Das zeigen Projektionen des Bundesamtes für Sozialversicherungen zur möglichen langfristigen Entwicklung der  AHV-Finanzen.3                                             |                                                     |
            | Auswirkungen  auf die IV                    | Die Initiative hat auch Auswirkungen auf die Invaliden                                                     |
            | versicherung. IV-Rentnerinnen und -Rentner erhalten eine  AHV-Rente, sobald sie das Rentenalter erreichen. Könnten sie  die AHV-Altersrente erst später beziehen, erhielten sie ihre  Rente entsprechend länger von der IV. Im Jahr 2033, wenn das  Rentenalter 66 erreicht wäre, würde das in der IV zu zusätzlichen Kosten von jährlich rund 200 Millionen Franken führen.  Danach würden diese zusätzlichen Kosten mit jeder Erhöhung  des Rentenalters weiter ansteigen.                                             |                                                     |

            3 Bericht des BSV vom 25.04.2023 zuhanden der SGK-N zu den Auswirkungen der Renteninitiative auf die Finanzen der AHV bis 2050 ( parlament.ch > Geschäfts-Nr. 22.054 > öffentliche Kommissionsunterlagen)

            ## Argumente Initiativkomitee

            | Am 1. Januar 1948 wurden die ersten AHV-Renten ausbezahlt.  Rentnerinnen und Rentner können seither auf ein sicheres  Einkommen im Alter zählen. Heute - 76 Jahre später - ist die AHV  in finanzieller Schieflage. Immer weniger Erwerbstätige finanzieren die AHV von immer mehr Rentnern. Tun wir nichts, sind  die AHV-Renten in Gefahr. Die Renteninitiative entschärft  diese Gefahr - ohne Rentenkürzungen, ohne zusätzliche  Steuern und ohne weitere Verschuldung.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                                                          |
            |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
            | AHV-Renten sichern                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Seit 76 Jahren zahlt die AHV Monat für Monat zuverlässig |
            | AHV-Renten an Rentnerinnen und Rentner aus. Doch nun steht  sie vor drei grossen Herausforderungen: Wir leben immer  länger, die Geburtenrate sinkt und in den nächsten zehn Jahren  werden über eine Million Erwerbstätige der sogenannten  Babyboomer-Generation pensioniert. Die Folge: Immer weniger  Erwerbstätige finanzieren die AHV-Renten von immer mehr  Rentnern. Tun wir nichts, sind die AHV-Renten gefährdet. Die  Renteninitiative wirkt dieser Entwicklung entgegen und stellt  die AHV-Finanzen wieder auf eine nachhaltige Basis. Davon  profitieren aktuelle und künftige Rentnerinnen und Rentner -  also unsere Kinder und Enkel. |                                                          |
            | Eine faire und  langfristige Lösung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Weil wir immer älter werden, müssen wir zwingend         |
            | etwas tun. Ohne Gegensteuer drohen Mehrwertsteuererhöhungen, mehr Lohnabgaben oder eine höhere Verschuldung.  Die beste Lösung, um die Renten nachhaltig zu sichern, bietet  die Renteninitiative. Eine moderate Verknüpfung des Rentenalters mit der steigenden Lebenserwartung ist fair für alle  Generationen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                                                          |

            Was aber ist mit Personen, die in körperlich beschwerlichen Berufen arbeiten? Wir unterstützen Branchenlösungen, wie es sie heute im Bau gibt, wo Bauarbeiter bereits früher in Pension gehen können.

            | Im weltweiten  Vergleich moderat                                                                                                                                                                                                                                                                                                                                 | Die Initiative ist moderat - besonders im internationalen   |
            |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
            | Vergleich. Dänemark, die Niederlande, Belgien, Deutschland  und viele weitere Staaten haben beschlossen, das Rentenalter  innerhalb der nächsten 10 Jahre auf 67 oder mehr zu erhöhen.  Mit der Renteninitiative steigt das Schweizer Rentenalter  hingegen nur auf 66 Jahre bis ins Jahr 2033. Das Anliegen der  Renteninitiative ist somit moderat und sozial. |                                                             |
            | Weniger  Zuwanderung                                                                                                                                                                                                                                                                                                                                             | Ein zusätzlicher Effekt der Renteninitiative: Sie reduziert |
            | die Zuwanderung in die Schweiz. Laut einer Studie im Auftrag  des Bundes kann die Renteninitiative zu einem Rückgang der  Zuwanderung in den Arbeitsmarkt um bis zu 23 Prozent bis ins  Jahr 2050 führen. Der Grund: Arbeitgeber können verstärkt  auf inländische Fachkräfte zurückgreifen.                                                                     |                                                             |

            Empfehlung des Initiativkomitees Darum empfiehlt das Initiativkomitee:
            Ja renten-sichern.ch Der Text auf dieser Doppelseite stammt vom Initiativkomitee. Es ist für den Inhalt und die Wortwahl verantwortlich.

            # Argumente Bundesrat Und Parlament

            | Ein Automatismus, der das Rentenalter an die Lebenserwartung  bindet, ist zu starr. Für Bundesrat und Parlament müssen bei  der Festlegung des Rentenalters stets verschiedene Aspekte  berücksichtigt werden, zum Beispiel auch die Entwicklung der  Wirtschaft, des Arbeitsmarktes und des Gesundheitszustands  der Bevölkerung. Das Rentenalter automatisch anhand einer  mathematischen Formel zu erhöhen, ohne diese Aspekte zu  beachten, ist zu einseitig. Bundesrat und Parlament lehnen die  Renteninitiative insbesondere aus folgenden Gründen ab:   |                                                         |
            |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
            | Automatismus  ist zu starr                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Mit der Initiative würde das Rentenalter künftig allein |
            | durch eine mathematische Formel bestimmt. Der in der Verfassung verankerte Automatismus würde greifen, egal wie die  Situation der älteren Arbeitnehmerinnen und Arbeitnehmer  aussieht. Das Rentenalter müsste erhöht werden, auch wenn  die Wirtschaft in einer Rezession steckt. Der Automatismus  liesse es nicht zu, andere Faktoren zu berücksichtigen oder das  Rentenalter langsamer oder gar nicht anzupassen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                                         |
            | Rentenalter 65  ist noch nicht  umgesetzt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Das Rentenalter der Frauen wird bis 2028 auf 65 Jahre   |
            | erhöht. Dies hat das Volk mit der letzten AHV-Reform im  September 2022 beschlossen. Der Bundesrat hält es nicht für  angebracht, das Rentenalter bereits wieder anzuheben, noch  bevor die Erhöhung des Frauenrentenalters vollzogen ist.                                                                                                                                                                                                                                                                                                                      |                                                         |
            | Initiative ist  einseitig                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Bundesrat und Parlament teilen das Anliegen der Initian                                                         |
            | tinnen und Initianten, eine nachhaltige Lösung für die finanziellen Herausforderungen der AHV zu finden. Was die Initiative  vorschlägt, ist jedoch einseitig. Eine Erhöhung des Rentenalters  über 65 Jahre hinaus soll nicht isoliert erfolgen, sondern muss  zusammen mit anderen Massnahmen im Rahmen der nächsten  AHV-Reform diskutiert werden.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                                         |

            | Nächste Reform ist  bereits aufgegleist                                                                                                                                                                                                                                            | Mit zwei Reformen in den letzten fünf Jahren sind die   |
            |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
            | Finanzen der AHV bis zirka 2030 stabilisiert. Auch die nächste  Reform ist schon aufgegleist, um die Renten darüber hinaus zu  sichern: Der Bundesrat wird dem Parlament bis Ende 2026 eine  ausgewogene Vorlage zur Stabilisierung der AHV für die Jahre  nach 2030 unterbreiten. |                                                         |
            | Automatismus ist  unschweizerisch                                                                                                                                                                                                                                                  | Die Altersvorsorge muss den gesellschaftlichen Entwick                                                         |
            | lungen laufend angepasst werden. Über zentrale Fragen wie  die Höhe des Rentenalters muss in einer direkten Demokratie  ein dauernder politischer Dialog geführt werden. Mit einem  Automatismus soll jedoch die Frage des angemessenen Rentenalters der politischen Diskussion praktisch entzogen werden.  Dies entspricht nicht der politischen Tradition der Schweiz.                                                                                                                                                                                                                                                                                    |                                                         |
            | Empfehlung von  Bundesrat und  Parlament                                                                                                                                                                                                                                           | Aus all diesen Gründen empfehlen Bundesrat und Parla                                                         |
            | ment, die Volksinitiative «Für eine sichere und nachhaltige  Altersvorsorge (Renteninitiative)» abzulehnen.                                                                                                                                                                        |                                                         |

            ## Nein

            admin.ch/renteninitiative
            """},
    ],
    response_format=StructuredData,
)

message = completion.choices[0].message
if message.parsed:
    print(message.parsed)
else:
    print(message.refusal)

# todo: save it in a better way
# save the response to a file
with open("response.json", "w") as f:
    f.write(message.json())
