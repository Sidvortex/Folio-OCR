import React, { useState } from 'react';
import { Heart, Activity, Users, AlertCircle, TrendingUp, Info, Sparkles, ThumbsUp } from 'lucide-react';

const CVDScreeningTool = () => {
  const [formData, setFormData] = useState({
    age: 50,
    sex: 'male',
    systolic_bp: 120,
    diastolic_bp: 80,
    bmi: 25,
    smoking: 'no',
    physical_activity: 'moderate'
  });
  
  const [result, setResult] = useState(null);
  const [calculationKey, setCalculationKey] = useState(0);
  
 
  const calculateRisk = () => {
    setCalculationKey(prev => prev + 1);
    
    let riskScore = 0;
    let riskFactors = [];
    
    if (formData.age > 65) {
      riskScore += 3;
      riskFactors.push("Age > 65 years");
    } else if (formData.age > 55) {
      riskScore += 2;
      riskFactors.push("Age > 55 years");
    } else if (formData.age > 45) {
      riskScore += 1;
    }
    
    if (formData.sex === 'male' && formData.age > 45) {
      riskScore += 1;
      riskFactors.push("Male over 45");
    }
    
    if (formData.systolic_bp >= 140 || formData.diastolic_bp >= 90) {
      riskScore += 3;
      riskFactors.push("High blood pressure (Hypertension)");
    } else if (formData.systolic_bp >= 130 || formData.diastolic_bp >= 85) {
      riskScore += 2;
      riskFactors.push("Elevated blood pressure");
    } else if (formData.systolic_bp >= 120) {
      riskScore += 1;
    }
    
    if (formData.bmi >= 30) {
      riskScore += 2;
      riskFactors.push("Obesity (BMI ≥ 30)");
    } else if (formData.bmi >= 25) {
      riskScore += 1;
      riskFactors.push("Overweight (BMI 25-30)");
    }
    

    if (formData.smoking === 'yes') {
      riskScore += 3;
      riskFactors.push("Current smoker");
    }
    

    if (formData.physical_activity === 'sedentary') {
      riskScore += 2;
      riskFactors.push("Sedentary lifestyle");
    } else if (formData.physical_activity === 'light') {
      riskScore += 1;
    }
    

    const probability = Math.min(95, Math.max(5, 15 + riskScore * 6.5));
    
    let riskLevel = 'Low';
    let riskColor = 'green';
    let recommendations = [];
    let healthTips = [];
    let funMessage = '';
    
    if (probability >= 60) {
      riskLevel = 'High';
      riskColor = 'red';
      recommendations = [
        "Immediate medical consultation recommended",
        "Blood pressure monitoring required",
        "Lifestyle modifications critical",
        "Consider preventive medication"
      ];
      healthTips = [
        "🩺 Schedule a comprehensive cardiovascular check-up within 1 week",
        "💊 Discuss blood pressure medication with your doctor",
        "🥗 Adopt a heart-healthy diet (DASH or Mediterranean diet)",
        "🚭 If smoking, seek cessation support immediately",
        "🏃‍♂️ Start with 10-minute daily walks, gradually increase",
        "😴 Aim for 7-8 hours of quality sleep per night",
        "🧘 Practice stress reduction (meditation, deep breathing)"
      ];
    } else if (probability >= 35) {
      riskLevel = 'Moderate';
      riskColor = 'orange';
      recommendations = [
        "Medical check-up within 1 month",
        "Monitor blood pressure regularly",
        "Increase physical activity",
        "Dietary modifications advised"
      ];
      healthTips = [
        "📅 Schedule a routine health check-up within 3-4 weeks",
        "🍎 Reduce sodium intake to <2,300mg per day",
        "🏋️ Aim for 150 minutes of moderate exercise weekly",
        "🥤 Limit alcohol consumption (men: 2 drinks/day, women: 1 drink/day)",
        "📊 Track your blood pressure at home weekly",
        "🥑 Increase intake of fruits, vegetables, and whole grains",
        "💧 Stay hydrated - drink 8 glasses of water daily"
      ];
    } else {
      riskLevel = 'Low';
      riskColor = 'green';
      recommendations = [
        "Maintain healthy lifestyle",
        "Annual health check-up",
        "Continue regular exercise",
        "Monitor weight and blood pressure"
      ];
      healthTips = [
        "Keep doing what you're doing! Your heart is vibing",
        "Stay consistent with your exercise routine",
        "Keep enjoying balanced, nutritious meals",
        "Annual check-ups are your friend - don't skip them",
        "Share your healthy habits with friends and family",
        "Consider becoming a fitness buddy for someone",
        "Your heart called - it says thanks for taking care of it! 💪"
      ];
      
      // Fun messages for low-risk patients
      const funMessages = [
        "Your heart is pumping like a champion! Keep slaying! 🔥",
        "Heart health on fleek! You're absolutely crushing it! 💯",
        "Your cardiovascular system said 'no worries fam!' 😎",
        "Heart game strong! You're basically a wellness guru! 🌟",
        "Your ticker is in beast mode! Respect! 👊",
        "Heart: *chef's kiss* Perfect! Keep this energy! ✨",
        "You're out here living your best heart-healthy life! Goals! 🎯"
      ];
      funMessage = funMessages[Math.floor(Math.random() * funMessages.length)];
    }
    
    setResult({
      probability: probability.toFixed(1),
      riskLevel,
      riskColor,
      riskFactors,
      recommendations,
      healthTips,
      funMessage,
      riskScore
    });
  };
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'age' || name === 'systolic_bp' || name === 'diastolic_bp' || name === 'bmi' 
        ? parseFloat(value) 
        : value
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-cyan-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <Heart className="w-12 h-12 text-red-500" />
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                PulseCheck: Lightweight AI for Early Cardiovascular Screening
              </h1>
              <p className="text-gray-600 mt-1">
                Accessible cardiovascular risk assessment for low-resource settings
              </p>
            </div>
          </div>
          
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mt-4">
            <div className="flex items-start">
              <Info className="w-5 h-5 text-blue-500 mt-0.5 mr-3 flex-shrink-0" />
              <div className="text-sm text-blue-800">
                <strong>Note:</strong> This is a screening tool for educational purposes and demonstration. 
                It does NOT replace professional medical diagnosis. Always consult healthcare providers for medical decisions.
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Form */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Users className="w-6 h-6" />
              Patient Information
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Age (years)
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  min="18"
                  max="100"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sex
                </label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Systolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    name="systolic_bp"
                    value={formData.systolic_bp}
                    onChange={handleInputChange}
                    min="80"
                    max="220"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Diastolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    name="diastolic_bp"
                    value={formData.diastolic_bp}
                    onChange={handleInputChange}
                    min="40"
                    max="140"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  BMI (kg/m²)
                </label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleInputChange}
                  min="15"
                  max="50"
                  step="0.1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Smoking Status
                </label>
                <select
                  name="smoking"
                  value={formData.smoking}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="no">Non-smoker</option>
                  <option value="yes">Current smoker</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Physical Activity Level
                </label>
                <select
                  name="physical_activity"
                  value={formData.physical_activity}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="sedentary">Sedentary (little to no exercise)</option>
                  <option value="light">Light (1-2 days/week)</option>
                  <option value="moderate">Moderate (3-5 days/week)</option>
                  <option value="active">Active (daily exercise)</option>
                </select>
              </div>

              <button
                onClick={calculateRisk}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-md transition-colors flex items-center justify-center gap-2"
              >
                <Activity className="w-5 h-5" />
                Calculate CVD Risk
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Risk Assessment Results
            </h2>

            {result ? (
              <div className="space-y-6" key={calculationKey}>
                {/* Risk Score Display */}
                <div className={`p-6 rounded-lg text-center bg-${result.riskColor}-50 border-2 border-${result.riskColor}-200`}>
                  <div className="text-5xl font-bold mb-2" style={{color: result.riskColor === 'red' ? '#ef4444' : result.riskColor === 'orange' ? '#f97316' : '#22c55e'}}>
                    {result.probability}%
                  </div>
                  <div className="text-lg font-semibold text-gray-700">
                    Estimated CVD Risk
                  </div>
                  <div className={`mt-2 inline-block px-4 py-1 rounded-full font-semibold bg-${result.riskColor}-100`} 
                       style={{color: result.riskColor === 'red' ? '#dc2626' : result.riskColor === 'orange' ? '#ea580c' : '#16a34a'}}>
                    {result.riskLevel} Risk
                  </div>
                </div>

                {/* Fun message for low risk */}
                {result.funMessage && (
                  <div className="bg-gradient-to-r from-green-100 to-emerald-100 border-2 border-green-400 rounded-lg p-4 text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Sparkles className="w-6 h-6 text-green-600" />
                      <ThumbsUp className="w-6 h-6 text-green-600" />
                    </div>
                    <p className="text-lg font-bold text-green-800">{result.funMessage}</p>
                  </div>
                )}

                {/* Risk Factors */}
                {result.riskFactors.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                      <AlertCircle className="w-5 h-5 text-orange-500" />
                      Identified Risk Factors
                    </h3>
                    <ul className="space-y-2">
                      {result.riskFactors.map((factor, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                          <span className="text-orange-500 mt-1">•</span>
                          <span>{factor}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommendations */}
                <div>
                  <h3 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Heart className="w-5 h-5 text-blue-500" />
                    Clinical Recommendations
                  </h3>
                  <ul className="space-y-2">
                    {result.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                        <span className="text-blue-500 mt-1">✓</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Health Improvement Tips */}
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 border-l-4 border-purple-500 p-4 rounded">
                  <h3 className="font-semibold text-purple-800 mb-2 flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    {result.riskLevel === 'Low' ? 'Keep It Up! 💪' : 'Action Steps to Improve Your Heart Health'}
                  </h3>
                  <ul className="space-y-2">
                    {result.healthTips.map((tip, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-purple-900">
                        <span className="text-purple-500 mt-1">→</span>
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                  <p className="text-sm text-yellow-800">
                    <strong>Important:</strong> This screening tool uses basic clinical parameters and 
                    simplified risk calculations. Professional medical evaluation with complete 
                    lab work and clinical examination is essential for accurate diagnosis.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Activity className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>Enter patient information and click "Calculate CVD Risk" to see results</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer Info */}
        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h3 className="font-bold text-gray-800 mb-3">About This Tool</h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Features Used</h4>
              <ul className="space-y-1">
                <li>• Age</li>
                <li>• Sex</li>
                <li>• Blood Pressure</li>
                <li>• BMI</li>
                <li>• Smoking Status</li>
                <li>• Physical Activity</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Advantages</h4>
              <ul className="space-y-1">
                <li>• No lab tests required</li>
                <li>• Low-cost screening</li>
                <li>• Quick assessment</li>
                <li>• Accessible anywhere</li>
                <li>• Interpretable results</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Limitations</h4>
              <ul className="space-y-1">
                <li>• Screening tool only</li>
                <li>• Not a diagnosis</li>
                <li>• Simplified model</li>
                <li>• Requires validation</li>
                <li>• Clinical oversight needed</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Creator Credit */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg shadow-lg p-6 mt-6 border-2 border-indigo-200">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Heart className="w-6 h-6 text-indigo-600" />
            <h3 className="font-bold text-gray-800 text-lg">Created By</h3>
            <Heart className="w-6 h-6 text-indigo-600" />
          </div>
          <p className="text-center text-indigo-900 font-semibold text-xl mb-2">
            Yuan Ching
          </p>
          <p className="text-center text-gray-600 text-sm">
            Machine Learning Engineer & Computational Health Researcher
          </p>
          <p className="text-center text-gray-500 text-xs mt-2">
            Dedicated to democratizing healthcare through accessible AI solutions
          </p>
        </div>
      </div>
    </div>
  );
};

export default CVDScreeningTool;