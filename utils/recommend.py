def get_recommendations(disease, age, sex, diabetes, hypertension):
    recs = []

    if disease == 'Normal':
        recs.append("Maintain regular eye check-ups every 6-12 months.")
        recs.append("Follow a balanced diet rich in fruits and leafy greens.")
        recs.append("Protect eyes from direct sunlight and digital strain.")

    if disease == 'Diabetic Retinopathy':
        recs.append("Monitor blood sugar levels daily.")
        recs.append("Follow a low-glycemic diet with whole grains and vegetables.")
        recs.append("Perform light eye exercises and maintain healthy weight.")
        if diabetes:
            recs.append("Ensure proper diabetes management and regular fundus exams.")

    if disease == 'Glaucoma':
        recs.append("Schedule frequent intraocular pressure (IOP) checks.")
        recs.append("Avoid smoking and manage stress to reduce eye pressure.")
        recs.append("Include antioxidant-rich foods such as spinach and kale.")
        if hypertension:
            recs.append("Maintain blood pressure below 120/80 through low-sodium diet.")

    if disease == 'Cataract':
        recs.append("Wear UV-protective sunglasses outdoors.")
        recs.append("Consume foods high in vitamins C and E.")
        recs.append("Avoid smoking and excessive alcohol consumption.")

    if age:
        try:
            age = int(age)
            if age > 50:
                recs.append("Include omega-3 fatty acids for better retinal health.")
        except:
            pass

    return "\n".join(recs)
