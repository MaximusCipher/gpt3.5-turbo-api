import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Define your dataset
dataset = [
    ("An employee was passed over for a promotion despite having more experience and qualifications than the person who was promoted.", "discrimination"),
    ("A company refused to hire an applicant because of their age, even though they were qualified for the position.", "age discrimination"),
    ("An employer made derogatory comments about a worker's disability and refused to provide reasonable accommodations.", "disability discrimination"),
    ("A pregnant employee was demoted and given fewer responsibilities because the employer assumed she wouldn't be able to handle the workload.", "pregnancy discrimination"),
    ("An employee was harassed by their supervisor because of their sexual orientation, leading to a hostile work environment.", "sexual orientation discrimination"),
    ("A transgender employee was denied access to the restroom that aligns with their gender identity, causing distress and discomfort.", "gender identity discrimination"),
    ("An employee was unfairly terminated after reporting instances of racial discrimination in the workplace.", "racial discrimination"),
    ("A worker was paid less than their colleagues for doing the same job because of their gender.", "gender discrimination"),
    ("An employee with a religious head covering was told they could not wear it at work, despite it being part of their religious practice.", "religious discrimination"),
    ("A Filipino worker was subjected to derogatory remarks and stereotypes about their nationality by coworkers.", "national origin discrimination"),
    ("An employer implemented a policy that disproportionately affected employees with disabilities, making it difficult for them to perform their jobs.", "disability discrimination"),
    ("A company refused to provide sign language interpreters for deaf employees during meetings and training sessions.", "disability discrimination"),
    ("An employee was denied a promotion because they requested time off for medical treatment related to a chronic illness.", "disability discrimination"),
    ("A pregnant worker was denied access to maternity leave and was forced to continue working despite experiencing complications.", "pregnancy discrimination"),
    ("An employee was subjected to unwanted advances and inappropriate comments by a supervisor, creating a hostile work environment.", "sexual harassment"),
    ("A transgender employee was repeatedly misgendered and called derogatory slurs by coworkers.", "gender identity discrimination"),
    ("An employee was excluded from team activities and social events because of their race.", "racial discrimination"),
    ("A worker was denied a job opportunity because they did not conform to traditional gender norms.", "gender discrimination"),
    ("An employer refused to accommodate an employee's religious practices and scheduled mandatory meetings during religious holidays.", "religious discrimination"),
    ("A Filipino worker was subjected to offensive jokes and comments about their accent and cultural background.", "national origin discrimination"),
    ("An employee with a visible disability was mocked and ridiculed by coworkers, creating a hostile work environment.", "disability discrimination"),
    ("A company implemented a policy requiring employees to undergo unnecessary medical examinations that targeted individuals with certain health conditions.", "disability discrimination"),
    ("An employee was denied access to training and development opportunities because they used a wheelchair.", "disability discrimination"),
    ("A pregnant worker was denied access to restroom breaks and water during long shifts, leading to health complications.", "pregnancy discrimination"),
    ("An employee was subjected to lewd comments and gestures by a coworker, despite repeatedly asking them to stop.", "sexual harassment"),
    ("A transgender employee was denied access to gender-affirming healthcare coverage by their employer's insurance plan.", "gender identity discrimination"),
    ("An employee of Filipino descent was unfairly disciplined for minor infractions while their non-Filipino coworkers received leniency for similar behavior.", "national origin discrimination"),
    ("A worker was repeatedly passed over for promotions because they did not fit the company's stereotypical image of leadership.", "gender discrimination"),
    ("An employer terminated an employee after they requested reasonable accommodations for their religious practices.", "religious discrimination"),
    ("A Filipino employee was subjected to microaggressions and subtle forms of discrimination by coworkers and supervisors.", "national origin discrimination"),
    ("An employee with a mental health condition was mocked and belittled by coworkers, contributing to their declining mental health.", "disability discrimination"),
    ("A company refused to provide necessary assistive technology and tools for an employee with a disability to perform their job effectively.", "disability discrimination"),
    ("An employee was denied promotion opportunities because they were perceived as being too old to learn new skills.", "age discrimination"),
    ("A transgender worker was denied access to restroom facilities at their workplace, causing distress and embarrassment.", "gender identity discrimination"),
    ("An employee was subjected to derogatory comments and jokes about their race and ethnicity by coworkers.", "racial discrimination"),
    ("A pregnant employee was denied opportunities for advancement and challenging assignments because of assumptions about their commitment to their career.", "pregnancy discrimination"),
    ("An employer implemented a dress code policy that targeted specific religious attire worn by employees.", "religious discrimination"),
    ("A Filipino worker was excluded from team meetings and decision-making processes by their non-Filipino coworkers.", "national origin discrimination"),
    ("An employee with a physical disability was denied access to essential workplace facilities and accommodations.", "disability discrimination"),
    ("A company refused to hire an applicant because of their marital status, believing that married individuals are less committed to their careers.", "marital status discrimination"),
    ("An employee was subjected to unwanted sexual advances and touching by a coworker, creating a hostile work environment.", "sexual harassment"),
    ("A transgender employee was denied access to appropriate restroom facilities and was forced to use facilities that did not align with their gender identity.", "gender identity discrimination"),
    ("An employee was unfairly disciplined for taking time off to observe religious holidays.", "religious discrimination"),
    ("A Filipino worker was denied opportunities for career advancement and professional development because of their ethnicity.", "national origin discrimination"),
    ("An employee with a mental health condition was subjected to stigmatizing remarks and assumptions about their abilities by coworkers and supervisors.", "disability discrimination"),
    ("A company refused to make accommodations for an employee with a disability, claiming it would be too costly and burdensome.", "disability discrimination"),
    ("An employer implemented a policy that required employees to work mandatory overtime shifts, despite knowing that it would negatively impact workers with caregiving responsibilities.", "caregiver discrimination"),
    ("A transgender employee was excluded from company-sponsored healthcare plans that covered gender-affirming medical treatments.", "gender identity discrimination"),
    ("An employee was unfairly terminated after filing a complaint about racial discrimination and harassment in the workplace.", "retaliation"),
    ("A pregnant worker was denied opportunities for career advancement and leadership roles due to stereotypes about pregnant individuals' commitment and abilities.", "pregnancy discrimination"),
    ("An employee was denied a promotion because they requested a flexible work schedule to accommodate their caregiving responsibilities.", "caregiver discrimination"),
    ("A transgender employee was refused access to gender-appropriate uniforms and was required to wear clothing that did not align with their gender identity.", "gender identity discrimination"),
    ("An employee was subjected to derogatory remarks and stereotypes about their age by younger coworkers.", "age discrimination"),
    ("A company refused to hire an applicant because they were perceived as being too young and lacking experience, despite meeting all the qualifications for the position.", "age discrimination"),
    ("An employee with a disability was denied access to training and professional development opportunities because of assumptions about their capabilities.", "disability discrimination"),
    ("A pregnant worker was denied access to restroom facilities and was told to 'hold it' by their supervisor, leading to discomfort and embarrassment.", "pregnancy discrimination"),
    ("An employer implemented a dress code policy that prohibited employees from wearing traditional cultural attire, forcing workers to conform to Western standards of dress.", "national origin discrimination"),
    ("A Filipino employee was excluded from team-building activities and networking events by their non-Filipino colleagues.", "national origin discrimination"),
    ("An employee with a mental health condition was denied a promotion because of assumptions about their ability to handle stress and pressure.", "disability discrimination"),
    ("A company refused to provide reasonable accommodations for an employee with a disability, citing concerns about cost and feasibility.", "disability discrimination"),
    ("An employee was denied access to career advancement opportunities because they did not conform to traditional gender norms and expectations.", "gender discrimination"),
    ("A transgender worker was excluded from participating in company-sponsored charity events and volunteer activities.", "gender identity discrimination"),
    ("An employee was subjected to unwelcome comments and jokes about their religious beliefs by coworkers, creating a hostile work environment.", "religious discrimination"),
    ("A company refused to make accommodations for an employee's religious practices, claiming it would disrupt business operations.", "religious discrimination"),
    ("An employee with a disability was denied access to a company-sponsored wellness program that required physical activity.", "disability discrimination"),
    ("A pregnant worker was denied access to company-provided transportation services, making it difficult for them to commute to work.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to use gender-segregated restrooms, disregarding the gender identities of transgender employees.", "gender identity discrimination"),
    ("An employee was excluded from company events and social gatherings because of their sexual orientation.", "sexual orientation discrimination"),
    ("A transgender employee was denied access to appropriate healthcare coverage for gender-affirming medical treatments by their employer's insurance plan.", "gender identity discrimination"),
    ("An employee was unfairly disciplined for taking time off to attend religious ceremonies and observances.", "religious discrimination"),
    ("A Filipino worker was mocked and belittled for their accent and language proficiency by coworkers.", "national origin discrimination"),
    ("An employee with a visible disability was refused service at a company-sponsored event because of accessibility issues.", "disability discrimination"),
    ("A company refused to provide sign language interpreters for deaf employees during mandatory training sessions.", "disability discrimination"),
    ("An employee was denied a promotion because they did not fit the company's stereotype of an ideal employee based on physical appearance.", "appearance discrimination"),
    ("A transgender worker was misgendered and referred to by their previous name by coworkers and supervisors, despite their request for respectful treatment.", "gender identity discrimination"),
    ("An employee was terminated after filing a complaint about sexual harassment in the workplace.", "retaliation"),
    ("A pregnant worker was denied access to job assignments and responsibilities that would have allowed for career growth and development.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to provide documentation of their marital status for family-related benefits, leading to discrimination against unmarried individuals.", "marital status discrimination"),
    ("An employee with a mental health condition was unfairly disciplined for taking breaks to manage their symptoms, despite providing documentation from a healthcare professional.", "disability discrimination"),
    ("A company refused to hire an applicant because they were a single parent, assuming that they would be unreliable and unable to commit to the job.", "caregiver discrimination"),
    ("An employee was denied access to training opportunities because they had taken extended leave for family caregiving responsibilities.", "caregiver discrimination"),
    ("A transgender worker was denied access to appropriate restroom facilities and was forced to use facilities that did not align with their gender identity.", "gender identity discrimination"),
    ("An employee was unfairly terminated after requesting time off to observe a religious holiday.", "religious discrimination"),
    ("A Filipino worker was denied access to training and development opportunities because of their accent and language proficiency.", "national origin discrimination"),
    ("An employee with a visible disability was excluded from participating in company-sponsored team-building activities and outdoor events.", "disability discrimination"),
    ("A company refused to provide reasonable accommodations for an employee with a disability, claiming it would create an undue burden on the organization.", "disability discrimination"),
    ("An employee was denied access to career advancement opportunities because they did not conform to traditional gender stereotypes and expectations.", "gender discrimination"),
    ("A transgender worker was excluded from participating in company-sponsored charity events and volunteer activities.", "gender identity discrimination"),
    ("An employee was subjected to unwelcome comments and jokes about their religious beliefs by coworkers, creating a hostile work environment.", "religious discrimination"),
    ("A company refused to make accommodations for an employee's religious practices, claiming it would disrupt business operations.", "religious discrimination"),
    ("An employee with a disability was denied access to a company-sponsored wellness program that required physical activity.", "disability discrimination"),
    ("A pregnant worker was denied access to company-provided transportation services, making it difficult for them to commute to work.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to use gender-segregated restrooms, disregarding the gender identities of transgender employees.", "gender identity discrimination"),
    ("An employee was excluded from company events and social gatherings because of their sexual orientation.", "sexual orientation discrimination"),
    ("A transgender employee was denied access to appropriate healthcare coverage for gender-affirming medical treatments by their employer's insurance plan.", "gender identity discrimination"),
    ("An employee was unfairly disciplined for taking time off to attend religious ceremonies and observances.", "religious discrimination"),
    ("A Filipino worker was mocked and belittled for their accent and language proficiency by coworkers.", "national origin discrimination"),
    ("An employee with a visible disability was refused service at a company-sponsored event because of accessibility issues.", "disability discrimination"),
    ("A company refused to provide sign language interpreters for deaf employees during mandatory training sessions.", "disability discrimination"),
    ("An employee was denied a promotion because they did not fit the company's stereotype of an ideal employee based on physical appearance.", "appearance discrimination"),
    ("A transgender worker was misgendered and referred to by their previous name by coworkers and supervisors, despite their request for respectful treatment.", "gender identity discrimination"),
    ("An employee was terminated after filing a complaint about sexual harassment in the workplace.", "retaliation"),
    ("A pregnant worker was denied access to job assignments and responsibilities that would have allowed for career growth and development.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to provide documentation of their marital status for family-related benefits, leading to discrimination against unmarried individuals.", "marital status discrimination"),
    ("An employee with a mental health condition was unfairly disciplined for taking breaks to manage their symptoms, despite providing documentation from a healthcare professional.", "disability discrimination"),
    ("A company refused to hire an applicant because they were a single parent, assuming that they would be unreliable and unable to commit to the job.", "caregiver discrimination"),
    ("An employee was denied access to training opportunities because they had taken extended leave for family caregiving responsibilities.", "caregiver discrimination"),
    ("A transgender worker was denied access to appropriate restroom facilities and was forced to use facilities that did not align with their gender identity.", "gender identity discrimination"),
    ("An employee was unfairly terminated after requesting time off to observe a religious holiday.", "religious discrimination"),
    ("A Filipino worker was denied access to training and development opportunities because of their accent and language proficiency.", "national origin discrimination"),
    ("An employee with a visible disability was excluded from participating in company-sponsored team-building activities and outdoor events.", "disability discrimination"),
    ("A company refused to provide reasonable accommodations for an employee with a disability, claiming it would create an undue burden on the organization.", "disability discrimination"),
    ("An employee was denied access to career advancement opportunities because they did not conform to traditional gender stereotypes and expectations.", "gender discrimination"),
    ("A transgender worker was excluded from participating in company-sponsored charity events and volunteer activities.", "gender identity discrimination"),
    ("An employee was subjected to unwelcome comments and jokes about their religious beliefs by coworkers, creating a hostile work environment.", "religious discrimination"),
    ("A company refused to make accommodations for an employee's religious practices, claiming it would disrupt business operations.", "religious discrimination"),
    ("An employee with a disability was denied access to a company-sponsored wellness program that required physical activity.", "disability discrimination"),
    ("A pregnant worker was denied access to company-provided transportation services, making it difficult for them to commute to work.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to use gender-segregated restrooms, disregarding the gender identities of transgender employees.", "gender identity discrimination"),
    ("An employee was excluded from company events and social gatherings because of their sexual orientation.", "sexual orientation discrimination"),
    ("A transgender employee was denied access to appropriate healthcare coverage for gender-affirming medical treatments by their employer's insurance plan.", "gender identity discrimination"),
    ("An employee was unfairly disciplined for taking time off to attend religious ceremonies and observances.", "religious discrimination"),
    ("A Filipino worker was mocked and belittled for their accent and language proficiency by coworkers.", "national origin discrimination"),
    ("An employee with a visible disability was refused service at a company-sponsored event because of accessibility issues.", "disability discrimination"),
    ("A company refused to provide sign language interpreters for deaf employees during mandatory training sessions.", "disability discrimination"),
    ("An employee was denied a promotion because they did not fit the company's stereotype of an ideal employee based on physical appearance.", "appearance discrimination"),
    ("A transgender worker was misgendered and referred to by their previous name by coworkers and supervisors, despite their request for respectful treatment.", "gender identity discrimination"),
    ("An employee was terminated after filing a complaint about sexual harassment in the workplace.", "retaliation"),
    ("A pregnant worker was denied access to job assignments and responsibilities that would have allowed for career growth and development.", "pregnancy discrimination"),
    ("An employer implemented a policy requiring employees to provide documentation of their marital status for family-related benefits, leading to discrimination against unmarried individuals.", "marital status discrimination"),
    ("An employee with a mental health condition was unfairly disciplined for taking breaks to manage their symptoms, despite providing documentation from a healthcare professional.", "disability discrimination"),
    ("A company refused to hire an applicant because they were a single parent, assuming that they would be unreliable and unable to commit to the job.", "caregiver discrimination"),
    ("An employee was denied access to training opportunities because they had taken extended leave for family caregiving responsibilities.", "caregiver discrimination")
]

X = [data[0] for data in dataset]
y = [data[1] for data in dataset]

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Initialize and Train Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_tfidf, y)

# Save the trained classifier and vectorizer
joblib.dump(classifier, 'classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
