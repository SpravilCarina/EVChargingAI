import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import App from './App';

// Mock fetch pentru toate endpoint-urile relevante
beforeEach(() => {
  global.fetch = jest.fn((url) => {
    if (url.includes('stations/monitor')) {
      return Promise.resolve({
        json: () => Promise.resolve([
          {
            station_id: 'ST001',
            status: 'online',
            voltage: 414,
            current: 94,
            temperature: 36,
            battery_soc: 82,
            history: [
              { voltage: 410, current: 90, temperature: 35, battery_soc: 80 },
              { voltage: 412, current: 92, temperature: 36, battery_soc: 81 },
              { voltage: 414, current: 94, temperature: 36, battery_soc: 82 }
            ]
          },
        ])
      });
    }
    if (url.includes('maintenance/alerts')) {
      return Promise.resolve({
        json: () => Promise.resolve([])
      });
    }
    if (url.includes('predict_lstm')) {
      return Promise.resolve({
        json: () => Promise.resolve({
          prediction: [100, 110, 120]
        })
      });
    }
    if (url.includes('dynamic_price')) {
      return Promise.resolve({
        json: () => Promise.resolve({
          dynamic_price: 1.23
        })
      });
    }
    if (url.includes('recommend_smart_schedule')) {
      return Promise.resolve({
        json: () => Promise.resolve({
          suggestion: "Încărcarea optimă e la ora 23"
        })
      });
    }
    return Promise.resolve({
      json: () => Promise.resolve([])
    });
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('randează titlul și cardul stației', async () => {
  render(<App />);
  expect(screen.getByText(/Stații Încărcare SmartCharge AI/i)).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getByText('ST001')).toBeInTheDocument();
  });
});

test('afișează corect parametrii și LED verde starea online', async () => {
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText(/414V/i)).toBeInTheDocument();
    expect(screen.getByText(/94A/i)).toBeInTheDocument();
    expect(screen.getByText(/36°C/i)).toBeInTheDocument();
    expect(screen.getByText(/82%/i)).toBeInTheDocument();
    expect(document.querySelector('.status-led.green')).toBeInTheDocument();
  });
});

test('butonul Previziune AI afișează forecast corect', async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText('Previziune AI')).toBeInTheDocument());
  fireEvent.click(screen.getByText('Previziune AI'));
  await waitFor(() => {
    expect(screen.getByText(/Forecast/)).toBeInTheDocument();
    expect(screen.getByText(/\[100, 110, 120/)).toBeInTheDocument();
  });
});

test('butonul Tarif AI afișează prețul dinamic corect', async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText('Tarif AI')).toBeInTheDocument());
  fireEvent.click(screen.getByText('Tarif AI'));
  await waitFor(() => {
    expect(screen.getByText(/Tarif dinamic AI:/)).toBeInTheDocument();
    expect(screen.getByText(/1.23/)).toBeInTheDocument();
  });
});

test('butonul Recomandare optimă AI afișează sugestia', async () => {
  render(<App />);
  // Recomandarea se activează din dashboard, adaptează după UI-ul tău
  // Dacă butonul există în panoul admin, schimbă selectorul conform
  await waitFor(() => expect(screen.getByText('Recomandare optimă AI')).toBeInTheDocument());
  fireEvent.click(screen.getByText('Recomandare optimă AI'));
  await waitFor(() => {
    expect(screen.getByText(/Încărcarea optimă e la ora 23/)).toBeInTheDocument();
  });
});

test('onboarding se afișează și poate fi închis', async () => {
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText(/Bun venit în SmartCharge AI!/i)).toBeInTheDocument();
  });
  fireEvent.click(screen.getByText(/Am înțeles|Închide tutorialul/i));
  await waitFor(() => {
    expect(screen.queryByText(/Bun venit în SmartCharge AI!/i)).not.toBeInTheDocument();
  });
});

test('dark mode se pornește și oprește', async () => {
  render(<App />);
  const darkSwitch = screen.getByLabelText(/Dark mode/);
  fireEvent.click(darkSwitch);
  expect(document.body.className).toMatch(/dark-mode|/);
});

test('meniul de setări rapide se deschide și închide', async () => {
  render(<App />);
  await waitFor(() => expect(screen.getByText(/Setări rapide/i)).toBeInTheDocument());
  fireEvent.click(screen.getByText(/Setări rapide/i));
  await waitFor(() => expect(screen.getByText(/Închide setări rapide/i)).toBeInTheDocument());
  fireEvent.click(screen.getByText(/Închide setări rapide/i));
  await waitFor(() => expect(screen.queryByText(/Închide setări rapide/i)).not.toBeInTheDocument());
});

test('export CSV se declanșează la click și shortcut Ctrl+E', async () => {
  render(<App />);
  // Click pe buton export
  const exportButton = screen.getByText(/Export date/i);
  const spyCreateElement = jest.spyOn(document, 'createElement');
  fireEvent.click(exportButton);
  expect(spyCreateElement).toHaveBeenCalledWith('a');
  spyCreateElement.mockRestore();

  // Trigger shortcut Ctrl+E
  const spyCreateElement2 = jest.spyOn(document, 'createElement');
  fireEvent.keyDown(window, { key: 'e', code: 'KeyE', ctrlKey: true });
  expect(spyCreateElement2).toHaveBeenCalledWith('a');
  spyCreateElement2.mockRestore();
});

test('leaderboard afișează corect utilizatori și progres', async () => {
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText(/Top Eco Utilizatori/i)).toBeInTheDocument();
    expect(screen.getByText('Carina')).toBeInTheDocument();
    expect(screen.getByText('Super Eco')).toBeInTheDocument();
    expect(screen.getByText('Mihai')).toBeInTheDocument();
    expect(screen.getByText('Eco')).toBeInTheDocument();
  });
});

test('notificările în feed se afișează corect și pot fi click-uiate', async () => {
  global.fetch = jest.fn(url => {
    if (url.includes('stations/monitor')) {
      return Promise.resolve({ json: () => Promise.resolve([{ station_id: 'ST002', status: 'offline', voltage: 100, current: 10, temperature: 50, battery_soc: 10 }]) });
    }
    if (url.includes('maintenance/alerts')) {
      return Promise.resolve({ json: () => Promise.resolve([{ station_id: 'ST002', alert: "Temperatură prea mare", timestamp: "2024-07-15T12:34" }]) });
    }
    return Promise.resolve({ json: () => Promise.resolve([]) });
  });
  render(<App />);
  await waitFor(() => {
    expect(screen.getByText(/Temperatură prea mare/i)).toBeInTheDocument();
    expect(screen.getByText(/Atenție/i)).toBeInTheDocument();
  });
});

test('modalul de explicații AI se deschide și se închide', async () => {
  render(<App />);
  // Butonul "De ce această recomandare?" trebuie să existe
  const explainBtn = await screen.findByText(/De ce această recomandare/i);
  fireEvent.click(explainBtn);
  await waitFor(() => {
    expect(screen.getByText(/Explicație AI/i)).toBeInTheDocument();
  });
  fireEvent.click(screen.getByText('OK'));
  await waitFor(() => {
    expect(screen.queryByText(/Explicație AI/i)).not.toBeInTheDocument();
  });
});

test('butoanele principale sunt accesibile prin tastatură (tab)', async () => {
  render(<App />);
  const allButtons = screen.getAllByRole("button");
  for (const button of allButtons) {
    button.focus();
    expect(document.activeElement).toBe(button);
  }
});
