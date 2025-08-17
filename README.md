# Probabilistic Methods for Data Quality in Healthcare
*A pragmatic assessment of integrating AI models for detecting data quality issues in healthcare.*

## Table of Contents
- [Overview](#Overview)
- [Use Cases](#Use-Cases)
- [Background](#Background)
    - [Why Data Quality Matters More Than Ever](#Why-Data-Quality-Matters-More-Than-Ever)
    - [Data Quality Defined](#Data-Quality-Defined)
    - [Knowledge Frameworks](#Knowledge-Frameworks)
    - [Data Quality Measurement](#Data-Quality-Measurement)
- [Overview](#Overview)
- [Overview](#Overview)

## Summary
Let's explore how **probabilistic methods and AI-driven anomaly detection** can enhance traditional data quality frameworks, making them more responsive, scalable, and effective in real-world healthcare environments. We'll briefly cover why organizations should care, what makes a robust data quality program, and examine a few AI-enabled probabilistic models suitable for data quality programs.

[TODO: List assessment criteria]

[TODO: List AI methods, findings, conclusion]

## Use Cases
In order to 

## Background

### Why Data Quality Matters More Than Ever
Data quality improvement programs have existed for ages but have mostly lacked the necessary incentive models for any meaningful impact. Data which was perfectly fine for clinical use by a human is not suitable for advanced digital processing. The growing use of healthcare data for interoperability, decision support, and quality measurement has raised both the standards for data quality and the burden of meeting them. Each of these applications are wholly dependent on the quality of the data itself to ensure system accuracy. In addition, more problems are being addressed with Artificial Intelligence (AI) which is well-known to be both data hungry and limited to the accuracy of its training data (i.e. training on bad data produces bad models). As a result, incentives and investments in data quality innovation will continue to grow alongside the exploding applications that need them.

### Data Quality Defined
If we are to improve data quality, we must first unambiguously define what it means (at least in the context of this post). Definitions of data quality are inherently nebulous, context specific, and strongly correlated to their use. Let's try and define this upfront.

> **Data Quality** is any issue present in or absence from the data which negatively impacts a given use case.

Exploring a few, obvious examples might help:
- A BMI recorded as `22.5 kg` instead of the correct `kg/m²`, conflating weight with a ratio.
- An A1c result of `6.4%` incorrectly coded as `Glucose [Mass/volume] in Blood` instead of `Glycated Hemoglobin`, misleading long-term glucose control assessments.
- An encounter date recorded in the future, which can distort care timelines and reporting.
- A telehealth service billed using HCPCS code `G2012` (Virtual check-in by a physician) which CMS stopped reimbursing after 2023.
- Bodysite missing for surgical procedures.
- No hypertension diagnosis for a large, aging population.

Other scenarios are more nuanced and case specific:
- A patient record contains only a zipcode or birth year, limiting the ability to stratify by age or region accurately.
- Prior authorization delays result in gaps between clinical recommendation and actual care delivery, affecting longitudinal data integrity.
- Latency in data synchronization (e.g., between EHR and analytics systems) leads to outdated or incomplete views of patient activity.
- Semantic mismatch or drift as data moves between systems or standards reducing accuracy, eroding trust, and decreasing usefulness.
- Exclusion data (like Frailty or Hospice) not fed to quality measurement system leading to lower clinical performance rates.

### Knowledge Frameworks
Good news for us, Data Quality is a well studied area. One of the leading frameworks for data quality applied to Healthcare is the [Kahn framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC5051581/). It classifies data quality [concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) into three distinct dimensions:
- **Conformance**: Mostly correct but coded or formatted incorrectly.
- **Completeness**: Absent because of capture, mapping, or operational issues (indeterminant based on what we know).
- **Plausibility**: Unusual or unlikely given contextual or clinical domain knowledge.

You might find scenarios which aren't cleanly classified using this taxonomy. For example, is timeliness a new data quality dimension or an operational quality issue which manifests under Completeness? For our purposes, we'll stick to the Kahn definitions and leave the semantics debates to academia.

### Data Quality Measurement
It's important to note that the healthcare data ecosystem is a highly dynamic and volitile environment where information is constantly being captured, exchanged, translated, enriched, and consumed across myriad organizational boundaries. More often than not, the organizations affected by poor data quality are not the ones positioned to fix them hence the critical importance of properly aligned incentive models and feedback loops. You see it all the time with data perfectly fine for clinical use by humans (e.g. reading a note) but insufficiently encoded for downstream automation (e.g. quality measurement).

Establishing robust data quality programs requires a [linqua franca](https://en.wikipedia.org/wiki/Lingua_franca) consumers and providers of quality data can use to assess, track, and improve quality over time. That begins with an accurate measurement for determining how well a bounded set of data meets the required data quality level for your program. A program might identify a set of key data quality measures, minimum targets for those measures, incentivize upstream data providers for meeting those targets, and feedback loops for when they don't.

## Additional Resources
- [ISO SQuaRE](https://www.iso.org/standard/35736.html)
- [Dimensions of Data Quality (DAMA)](https://dama-nl.org/dimensions-of-data-quality-en/)
- The [PIQI Framework](https://piqiframework.org) by Clinical Architecture